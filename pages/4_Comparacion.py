"""
Módulo 4 — Comparación de Métodos
=================================
Compara Markowitz (media-varianza), NSGA-II (GA), DP (proxy mínima varianza) y
Buy&Hold / Equiponderado sobre las mismas series. Produce tabla de métricas,
gráfico de barras, evolución de riqueza superpuesta, ranking automático y un
reporte Excel con múltiples hojas (una por método).

Parámetros base (tickers, fechas, capital) desde st.session_state.
"""

import io
import random
import warnings

import bootstrap  # noqa: F401 — debe ir antes de cualquier import de numpy/scipy (ver bootstrap.py)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from deap import base, creator, tools
import streamlit as st

from estilos import aplicar_estilos
from sidebar import renderizar_sidebar
from datos import descargar_precios, TTL_PRECIOS_SEGUNDOS
from finanzas import negative_sharpe_ratio, portfolio_performance, portfolio_volatility

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuración y paleta
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Comparación", page_icon="🏆", layout="wide")

# Estilos (paleta, tipografía, ajuste de modo oscuro y renombrado del sidebar)
#   Definidos en estilos.py para reutilizarse igual en todos los módulos.
aplicar_estilos()

DIAS_ANIO, RF, SEMILLA = 252, 0.02, 42

st.markdown(
    "<h1>🏆 Módulo 4 · Comparación de Métodos</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# SIDEBAR — Configuración de Parámetros (compartido con las demás páginas)
# --------------------------------------------------------------------------- #
parametros = renderizar_sidebar(detener_si_invalido=True)
tickers_lista = parametros["tickers_lista"]
fecha_ini = parametros["fecha_ini"]
fecha_fin = parametros["fecha_fin"]
capital = parametros["capital"]
MAX_CASH = parametros["max_cash"]
ejecutar = parametros["ejecutar"]
forzar_recalculo = parametros["forzar_recalculo"]

# Variables finales para usar en el resto del módulo
TICKERS = tickers_lista
FECHA_INICIO = str(fecha_ini)
FECHA_FIN = str(fecha_fin)
CAPITAL = float(capital)

st.caption(
    f"**Universo:** {', '.join(TICKERS)}  |  **Periodo:** {FECHA_INICIO} → {FECHA_FIN}  "
    f"|  **Capital:** ${CAPITAL:,.0f}  |  🛡️ **Límite Efectivo:** {MAX_CASH * 100:.0f}%"
)

if not TICKERS:
    # Salvaguarda extra: en teoría ya no debería llegar aquí, porque
    # renderizar_sidebar(detener_si_invalido=True) corta la ejecución antes
    # si no hay tickers válidos. Se deja por robustez ante ediciones futuras.
    st.error("⚠️ No hay tickers configurados. Vuelve al inicio y define el universo.")
    st.stop()

COLORES = ["#1F3864", "#4472C4", "#CC0000", "#FF6666", "#2E7D32", "#81C784", "gray"]
# --------------------------------------------------------------------------- #
# Cálculo de todos los métodos (cacheado por parámetros)
#   TTL: mismo TTL_PRECIOS_SEGUNDOS que descargar_precios() en datos.py,
#   para que la comparación completa no quede cacheada indefinidamente
#   más allá de lo que ya expira la descarga de precios en la que se basa.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=TTL_PRECIOS_SEGUNDOS)
def calcular_estrategias(tickers, inicio, fin, capital, max_cash,
                          reuse_markowitz=None, reuse_nsga2=None, reuse_dp=None):
    """Calcula (o reutiliza) los pesos de cada método y simula su evolución.

    reuse_markowitz / reuse_nsga2 / reuse_dp: dict {ticker: peso} ya
    calculado por los módulos 1/2/3 respectivamente para EXACTAMENTE los
    mismos parámetros (tickers, fechas, capital, límite de efectivo) — o
    None si no hay nada reutilizable. Cuando se provee un dict válido, se
    usa directamente en vez de re-resolver la optimización correspondiente.
    Esto es especialmente valioso para NSGA-II, la parte más costosa de
    este módulo (población de 80 individuos x 60 generaciones).
    """
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)

    precios, tickers_descartados = descargar_precios(tickers, inicio, fin)
    
    # 1. PREPARACIÓN E INYECCIÓN DE CASH
    retornos_log = np.log(precios / precios.shift(1)).dropna()
    ret_simples = precios.pct_change().dropna()
    
    # Inyectar rentabilidad del activo libre de riesgo
    retornos_log['CASH'] = RF / DIAS_ANIO
    ret_simples['CASH'] = RF / DIAS_ANIO
    
    tickers_validos = list(retornos_log.columns)
    N = len(tickers_validos)
    
    mu_vec = retornos_log.mean().values * DIAS_ANIO
    Sigma = retornos_log.cov().values * DIAS_ANIO
    
    # Límite global: Las acciones van de 0 a 1, CASH (último) tiene tope de max_cash
    limites_produccion = [(0.0, 1.0)] * (N - 1) + [(0.0, max_cash)]

    def _pesos_desde_reuse(reuse_dict):
        """Reconstruye el vector de pesos en el orden de `tickers_validos`
        a partir de un dict {ticker: peso} ya calculado, validando que
        contenga exactamente los mismos activos y sume ~1. Si algo no
        cuadra (p. ej. yfinance devolvió un universo de activos válidos
        distinto), retorna None para forzar el recálculo normal."""
        if not reuse_dict:
            return None
        try:
            w = np.array([reuse_dict[t] for t in tickers_validos], dtype=float)
        except KeyError:
            return None
        if w.shape[0] != N or not np.isclose(w.sum(), 1.0, atol=1e-3):
            return None
        return w

    # --- Markowitz: máximo Sharpe ---
    w_markowitz = _pesos_desde_reuse(reuse_markowitz)
    if w_markowitz is None:
        w_markowitz = minimize(negative_sharpe_ratio, np.ones(N) / N, args=(mu_vec, Sigma, RF),
                               method="SLSQP",
                               bounds=limites_produccion,
                               constraints={"type": "eq", "fun": lambda w: w.sum() - 1}).x

    # --- NSGA-II ---
    w_ga = _pesos_desde_reuse(reuse_nsga2)
    if w_ga is None:
        if hasattr(creator, "FitM4"):
            del creator.FitM4
        if hasattr(creator, "IndM4"):
            del creator.IndM4
        creator.create("FitM4", base.Fitness, weights=(-1.0, -1.0))
        creator.create("IndM4", list, fitness=creator.FitM4)

        def decodificar(ind):
            w = np.clip(np.array(ind, dtype=float), 0, None)
            s = w.sum()
            w = w / s if s > 1e-10 else np.ones(N) / N
            
            # Restricción de efectivo
            if w[-1] > max_cash:
                exceso = w[-1] - max_cash
                w[-1] = max_cash
                suma_acciones = w[:-1].sum()
                if suma_acciones > 1e-10:
                    w[:-1] += exceso * (w[:-1] / suma_acciones)
                else:
                    w[:-1] += exceso / (N - 1)
            return w

        tb = base.Toolbox()
        tb.register("attr_float", random.random)
        tb.register("individual", tools.initRepeat, creator.IndM4, tb.attr_float, n=N)
        tb.register("population", tools.initRepeat, list, tb.individual)

        def eval_ga(ind):
            w = decodificar(ind)
            ret, vol = portfolio_performance(w, mu_vec, Sigma)
            return (-ret, vol)

        tb.register("evaluate", eval_ga)
        tb.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
        tb.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1.0 / N)
        tb.register("select", tools.selNSGA2)

        pop = tb.population(n=80)
        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)
        pop = tb.select(pop, 80)
        for _ in range(60):
            off = [tb.clone(i) for i in tools.selTournamentDCD(pop, len(pop))]
            for i in range(0, len(off) - 1, 2):
                if random.random() < 0.9:
                    tb.mate(off[i], off[i + 1])
                    del off[i].fitness.values, off[i + 1].fitness.values
            for i in off:
                if random.random() < 0.2:
                    tb.mutate(i)
                    del i.fitness.values
            for i in [x for x in off if not x.fitness.valid]:
                i.fitness.values = tb.evaluate(i)
            pop = tb.select(pop + off, 80)

        frente = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        pts_ga = np.array([i.fitness.values for i in frente])
        pts_ga[:, 0] *= -1
        best_ga = int(np.argmax(pts_ga[:, 0] / pts_ga[:, 1]))

        # Decodificamos el mejor individuo para asegurar que respete los límites
        w_ga = decodificar(frente[best_ga])

    # --- DP: mínima varianza como proxy ---
    w_dp = _pesos_desde_reuse(reuse_dp)
    if w_dp is None:
        w_dp = minimize(portfolio_volatility, np.ones(N) / N, args=(mu_vec, Sigma), method="SLSQP",
                        bounds=limites_produccion,
                        constraints={"type": "eq", "fun": lambda w: w.sum() - 1}).x
                    
    # --- Equiponderado ---
    w_eq = np.ones(N) / N
    if w_eq[-1] > max_cash:
        exceso = w_eq[-1] - max_cash
        w_eq[-1] = max_cash
        w_eq[:-1] += exceso / (N - 1)

    # --- Simulación ---
    def simular(w_opt, rebalancear=False):
        riqueza = [capital]
        w_t = w_opt.copy()
        ult_mes = ret_simples.index[0].month
        for i in range(len(ret_simples)):
            r = ret_simples.iloc[i].values
            riqueza.append(riqueza[-1] * (1 + w_t @ r))
            if rebalancear and ret_simples.index[i].month != ult_mes:
                w_t = w_opt.copy()
                ult_mes = ret_simples.index[i].month
            else:
                w_t = w_t * (1 + r)
                w_t /= w_t.sum()
        return riqueza

    estrategias = {
        "Markowitz B&H": simular(w_markowitz, False),
        "Markowitz Rebal.": simular(w_markowitz, True),
        "NSGA-II B&H": simular(w_ga, False),
        "NSGA-II Rebal.": simular(w_ga, True),
        "DP (MínVar) B&H": simular(w_dp, False),
        "DP (MínVar) Rebal.": simular(w_dp, True),
        "Equiponderado": simular(w_eq, False),
    }
    fechas = [precios.index[0]] + list(ret_simples.index)
    pesos = {"Markowitz": dict(zip(tickers_validos, w_markowitz)),
             "NSGA-II": dict(zip(tickers_validos, w_ga)),
             "DP (MínVar)": dict(zip(tickers_validos, w_dp)),
             "Equiponderado": dict(zip(tickers_validos, w_eq))}
             
    return estrategias, [str(f.date()) for f in fechas], pesos, tickers_descartados


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metricas(riqueza, capital):
    serie = pd.Series(riqueza)
    rets = serie.pct_change().dropna()
    sharpe = (rets.mean() * DIAS_ANIO - RF) / (rets.std() * np.sqrt(DIAS_ANIO))
    dd = (serie.cummax() - serie) / serie.cummax()
    neg = rets[rets < 0]
    sortino = ((rets.mean() * DIAS_ANIO - RF) / (neg.std() * np.sqrt(DIAS_ANIO))
               if len(neg) > 0 else 0.0)
    return {
        "Riqueza Final": riqueza[-1],
        "Retorno Total %": (riqueza[-1] / capital - 1) * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown %": dd.max() * 100,
    }


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
with st.spinner("Calculando y comparando todos los métodos..."):
    # --------------------------------------------------------------------- #
    # Reutilización de resultados ya calculados en los módulos 1-3
    #   Si el usuario ya ejecutó Markowitz / NSGA-II / DP en sus respectivas
    #   páginas con EXACTAMENTE los mismos parámetros que tiene configurados
    #   ahora mismo (mismos tickers, fechas, capital y límite de efectivo),
    #   no tiene sentido recalcular todo desde cero aquí — sobre todo
    #   NSGA-II, que es la parte más costosa (80 individuos x 60
    #   generaciones). Se reutilizan sus pesos ya calculados; si algún
    #   parámetro cambió, esa parte se recalcula normalmente.
    # --------------------------------------------------------------------- #
    parametros_actuales = (tuple(TICKERS), FECHA_INICIO, FECHA_FIN, CAPITAL, MAX_CASH)

    reuse_markowitz = (
        st.session_state.get("markowitz_pesos")
        if st.session_state.get("markowitz_params") == parametros_actuales else None
    )
    reuse_nsga2 = (
        st.session_state.get("nsga2_pesos")
        if st.session_state.get("nsga2_params") == parametros_actuales else None
    )
    reuse_dp = (
        st.session_state.get("dp_pesos")
        if st.session_state.get("dp_params") == parametros_actuales else None
    )

    if forzar_recalculo:
        # "Forzar recálculo" implica no confiar en NADA cacheado — ni en la
        # caché propia de esta página, ni en los pesos reutilizados de los
        # módulos 1-3 (que también podrían estar "viejos" respecto a lo que
        # se busca verificar).
        descargar_precios.clear()
        calcular_estrategias.clear()
        reuse_markowitz = reuse_nsga2 = reuse_dp = None
        st.caption("🔄 Forzando recálculo completo (caché descartada, sin reutilizar módulos 1-3).")
    elif reuse_markowitz or reuse_nsga2 or reuse_dp:
        reutilizados = [n for n, r in (
            ("Markowitz", reuse_markowitz), ("NSGA-II", reuse_nsga2), ("DP", reuse_dp),
        ) if r]
        st.caption(f"♻️ Reutilizando resultados ya calculados de: {', '.join(reutilizados)}.")

    estrategias, fechas, pesos, tickers_descartados = calcular_estrategias(
        tuple(TICKERS), FECHA_INICIO, FECHA_FIN, CAPITAL, MAX_CASH,
        reuse_markowitz=reuse_markowitz, reuse_nsga2=reuse_nsga2, reuse_dp=reuse_dp,
    )

if tickers_descartados:
    st.warning(
        "⚠️ Se descartaron los siguientes tickers por no tener datos válidos en "
        f"Yahoo Finance para el rango de fechas seleccionado: {', '.join(tickers_descartados)}. "
        "La comparación continuó con el resto del universo."
    )

# Tabla de métricas
filas = []
for nombre, riq in estrategias.items():
    m = metricas(riq, CAPITAL)
    m = {"Método": nombre, **m}
    filas.append(m)
df_resumen = pd.DataFrame(filas)

# --------------------------------------------------------------------------- #
# Ranking automático
# --------------------------------------------------------------------------- #
mejor_sharpe = df_resumen.loc[df_resumen["Sharpe"].idxmax(), "Método"]
mejor_riqueza = df_resumen.loc[df_resumen["Riqueza Final"].idxmax(), "Método"]

c1, c2 = st.columns(2)
c1.metric("🥇 Mejor por Sharpe", mejor_sharpe,
          delta=f"{df_resumen['Sharpe'].max():.3f}")
c2.metric("💰 Mejor por Riqueza", mejor_riqueza,
          delta=f"${df_resumen['Riqueza Final'].max():,.0f}")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Tabla interactiva
# --------------------------------------------------------------------------- #
st.markdown("### Tabla resumen de métricas")
df_rank = df_resumen.sort_values("Sharpe", ascending=False).reset_index(drop=True)
df_rank.index += 1
df_rank.index.name = "Rank"
st.dataframe(
    df_rank.style.format({
        "Riqueza Final": "${:,.0f}",
        "Retorno Total %": "{:.1f}%",
        "Sharpe": "{:.4f}",
        "Sortino": "{:.4f}",
        "Max Drawdown %": "{:.1f}%",
    }).background_gradient(subset=["Sharpe"], cmap="Blues"),
    width='stretch',
)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Gráficos de barras comparativos (Sharpe, Riqueza y Max Drawdown)
# --------------------------------------------------------------------------- #
st.markdown("### Comparación de Métricas por Método")
col_b1, col_b2, col_b3 = st.columns(3)
nombres = df_resumen["Método"].tolist()

with col_b1:
    fig_s = go.Figure(go.Bar(
        x=df_resumen["Sharpe"], y=nombres, orientation="h",
        marker_color=COLORES, text=[f"{v:.3f}" for v in df_resumen["Sharpe"]],
        textposition="outside",
    ))
    fig_s.update_layout(title="Sharpe Ratio por método", height=380,
                        margin=dict(t=40, b=30, l=10, r=10),
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_s, width='stretch')

with col_b2:
    fig_r = go.Figure(go.Bar(
        x=df_resumen["Riqueza Final"], y=nombres, orientation="h",
        marker_color=COLORES,
        text=[f"${v:,.0f}" for v in df_resumen["Riqueza Final"]],
        textposition="outside",
    ))
    fig_r.update_layout(title="Riqueza final por método", height=380,
                        margin=dict(t=40, b=30, l=10, r=10),
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_r, width='stretch')

with col_b3:
    fig_d = go.Figure(go.Bar(
        x=df_resumen["Max Drawdown %"], y=nombres, orientation="h",
        marker_color=COLORES,
        text=[f"{v:.1f}%" for v in df_resumen["Max Drawdown %"]],
        textposition="outside",
    ))
    fig_d.update_layout(title="Max Drawdown % por método (menor es mejor)", height=380,
                        margin=dict(t=40, b=30, l=10, r=10),
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_d, width='stretch')

st.markdown("---")

# --------------------------------------------------------------------------- #
# Evolución de riqueza superpuesta (interactivo)
# --------------------------------------------------------------------------- #
st.markdown("### Evolución de la riqueza superpuesta")
fig_w = go.Figure()
for (nombre, riq), color in zip(estrategias.items(), COLORES):
    dash = "dash" if "Rebal" in nombre else "solid"
    fig_w.add_trace(go.Scatter(
        x=fechas, y=riq, mode="lines", name=f"{nombre} (${riq[-1]:,.0f})",
        line=dict(color=color, dash=dash, width=1.8),
    ))
fig_w.add_hline(y=CAPITAL, line=dict(color="black", dash="dot"), opacity=0.3)
fig_w.update_layout(
    xaxis_title="Fecha", yaxis_title="Valor del portafolio (USD)",
    height=520, legend=dict(font=dict(size=10)),
    margin=dict(t=20, b=40, l=40, r=20),
)
st.plotly_chart(fig_w, width='stretch')

st.markdown("---")

# --------------------------------------------------------------------------- #
# Descarga reporte Excel multi-hoja (una por método)
# --------------------------------------------------------------------------- #
st.markdown("### Reporte completo")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    # Hoja resumen
    df_export = df_resumen.copy()
    df_export.to_excel(writer, index=False, sheet_name="Resumen")
    # Hoja de pesos
    pd.DataFrame(pesos).to_excel(writer, sheet_name="Pesos")
    # Una hoja por estrategia con su serie de riqueza
    for nombre, riq in estrategias.items():
        hoja = nombre.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")[:31]
        pd.DataFrame({"Fecha": fechas, "Riqueza": riq}).to_excel(
            writer, index=False, sheet_name=hoja)
buffer.seek(0)

st.download_button(
    label="⬇️ Descargar reporte completo (Excel, multi-hoja)",
    data=buffer,
    file_name="reporte_comparacion_metodos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown(
    "<div class='disclaimer' style='margin-top:1rem'>⚠️ <b>Aviso:</b> Los datos son "
    "simulaciones con fines académicos y no constituyen asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
