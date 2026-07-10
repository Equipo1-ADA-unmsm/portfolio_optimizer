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
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from deap import base, creator, tools
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuración y paleta
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Comparación", page_icon="🏆", layout="wide")
AZUL, GRANATE, DORADO = "#1F3864", "#800000", "#C5961A"
DIAS_ANIO, RF, SEMILLA = 252, 0.02, 42

st.markdown(
    """
    <style>
        div[data-testid="stSidebarNav"] ul li:first-child a span {
            font-size: 0 !important;
        }
        div[data-testid="stSidebarNav"] ul li:first-child a span::before {
            content: "Dashboard Principal" !important;
            font-size: 14px !important;
            font-weight: 500;
            color: var(--text-color);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='color:{AZUL}'>🏆 Módulo 4 · Comparación de Métodos</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# NUEVO: SIDEBAR — Configuración de Parámetros
# --------------------------------------------------------------------------- #
TICKERS_DEFAULT = "FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"
FECHA_INI_DEFAULT = dt.date(2015, 1, 1)
FECHA_FIN_DEFAULT = dt.date(2024, 12, 31)
CAPITAL_DEFAULT = 100_000

with st.sidebar:
    st.markdown(f"<h2 style='color:{AZUL};margin-bottom:0'>⚙️ Parámetros</h2>",
                unsafe_allow_html=True)
    st.caption("Configuración global del análisis")

    tickers_input = st.text_input(
        "Tickers (separados por coma)",
        value=st.session_state.get("tickers_raw", TICKERS_DEFAULT),
        help="Símbolos de Yahoo Finance. Ej: FSM, BHP, BVN",
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fecha_ini = st.date_input(
            "Fecha inicio",
            value=st.session_state.get("fecha_ini", FECHA_INI_DEFAULT),
            min_value=dt.date(2000, 1, 1),
            max_value=dt.date.today(),
        )
    with col_f2:
        fecha_fin = st.date_input(
            "Fecha fin",
            value=st.session_state.get("fecha_fin", FECHA_FIN_DEFAULT),
            min_value=dt.date(2000, 1, 1),
            max_value=dt.date.today(),
        )

    capital = st.number_input(
        "Capital a invertir (USD)",
        min_value=1_000,
        max_value=100_000_000,
        value=st.session_state.get("capital", CAPITAL_DEFAULT),
        step=1_000,
        format="%d",
    )
    
    MAX_CASH = st.slider(
        "Límite máx. Efectivo", 
        0.0, 1.0, float(st.session_state.get("max_cash", 0.20)), 
        step=0.05, 
        format="%.2f"
    )

    st.markdown("---")
    ejecutar = st.button("🚀 Ejecutar Análisis")

    st.markdown("---")
    st.caption("💡 Los parámetros se comparten entre todas las páginas.")

# Sincronización y reactividad con session_state
tickers_lista = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.session_state["tickers_raw"] = tickers_input
st.session_state["tickers"] = tickers_lista
st.session_state["fecha_ini"] = fecha_ini
st.session_state["fecha_fin"] = fecha_fin
st.session_state["capital"] = int(capital)
st.session_state["max_cash"] = float(MAX_CASH)

if ejecutar:
    st.session_state["analisis_ejecutado"] = True

# Validaciones
if fecha_ini >= fecha_fin:
    st.sidebar.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
if not tickers_lista:
    st.sidebar.error("⚠️ Ingresa al menos un ticker.")
if capital <= 0:
    st.sidebar.error("⚠️ El capital debe ser mayor que 0.")

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
    st.error("⚠️ No hay tickers configurados. Vuelve al inicio y define el universo.")
    st.stop()

COLORES = ["#1F3864", "#4472C4", "#CC0000", "#FF6666", "#2E7D32", "#81C784", "gray"]
# --------------------------------------------------------------------------- #
# Datos (cacheado)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cargar_datos(tickers, inicio, fin):
    datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
    precios = datos["Close"]
    if isinstance(precios, pd.Series):
        precios = precios.to_frame()
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)
    return precios.dropna(how="all").dropna()


# --------------------------------------------------------------------------- #
# Cálculo de todos los métodos (cacheado por parámetros)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def calcular_estrategias(tickers, inicio, fin, capital, max_cash):
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)

    precios = cargar_datos(tickers, inicio, fin)
    
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

    # --- Markowitz: máximo Sharpe ---
    def neg_sharpe(w):
        # abs() para proteger de errores de precisión flotante con el CASH
        return -(w @ mu_vec - RF) / np.sqrt(abs(w @ Sigma @ w))
        
    w_markowitz = minimize(neg_sharpe, np.ones(N) / N, method="SLSQP",
                           bounds=limites_produccion,
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1}).x

    # --- NSGA-II ---
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
        return (-(w @ mu_vec), np.sqrt(abs(w @ Sigma @ w)))

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
    w_dp = minimize(lambda w: np.sqrt(abs(w @ Sigma @ w)), np.ones(N) / N, method="SLSQP",
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
             
    return estrategias, [str(f.date()) for f in fechas], pesos


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
    estrategias, fechas, pesos = calcular_estrategias(
        tuple(TICKERS), FECHA_INICIO, FECHA_FIN, CAPITAL, MAX_CASH
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
    use_container_width=True,
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
    st.plotly_chart(fig_s, use_container_width=True)

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
    st.plotly_chart(fig_r, use_container_width=True)

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
    st.plotly_chart(fig_d, use_container_width=True)

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
st.plotly_chart(fig_w, use_container_width=True)

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
    f"<div style='background:#FDF6E3;border-left:5px solid {DORADO};color:{GRANATE};"
    "padding:0.8rem 1rem;border-radius:6px;font-size:0.88rem;margin-top:1rem'>⚠️ "
    "<b>Aviso:</b> Los datos son simulaciones con fines académicos y no constituyen "
    "asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
