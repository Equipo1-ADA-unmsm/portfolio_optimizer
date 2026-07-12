"""
Módulo 3 — Programación Dinámica (Rebalanceo)
=============================================
Backward induction de Bellman para la decisión secuencial de rebalanceo de un
portafolio, penalizando costos de transacción y suboptimalidad respecto al
portafolio objetivo (mínima varianza). Compara 3 estrategias: Buy&Hold,
DP optimizado y Siempre-rebalanceado.

Parámetros base (tickers, fechas, capital) desde st.session_state.
λ_TC, T (periodos) y paso de grilla configurables con sliders.

Ref.: Vaezi Jezeie et al. (2022). PLoS ONE 17(8), e0271811.
"""

import io
import warnings
from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
from sidebar import renderizar_sidebar
from datos import descargar_precios

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuración y paleta
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="DP Rebalanceo", page_icon="🔁", layout="wide")

# Estilos (paleta, tipografía, ajuste de modo oscuro y renombrado del sidebar)
#   Definidos en estilos.py para reutilizarse igual en todos los módulos.
aplicar_estilos()

VERDE = "#2E7D32"  # Color adicional exclusivo de este módulo (estrategia "Siempre Rebalanceado")
DIAS_ANIO, RF = 252, 0.02

st.markdown(
    "<h1>🔁 Módulo 3 · Programación Dinámica (Rebalanceo)</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# SIDEBAR — Configuración de Parámetros (compartido con las demás páginas)
#   Sin botón genérico: este módulo tiene su propio botón "🔁 Ejecutar DP".
# --------------------------------------------------------------------------- #
parametros = renderizar_sidebar(mostrar_boton_ejecutar=False)
tickers_lista = parametros["tickers_lista"]
fecha_ini = parametros["fecha_ini"]
fecha_fin = parametros["fecha_fin"]
capital = parametros["capital"]
MAX_CASH = parametros["max_cash"]

# --------------------------------------------------------------------------- #
# Sliders del modelo DP
# --------------------------------------------------------------------------- #
col_s1, col_s2, col_s3, col_s4 = st.columns([2, 2, 2, 2])
with col_s1:
    LAMBDA_TC = st.slider("λ_TC (costo de transacción)", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
with col_s2:
    T_PERIODOS = st.slider("T (periodos de rebalanceo)", 4, 52, 12, step=1)
with col_s3:
    # Límite máximo ajustado a 10 para evitar que la "Maldición de la Dimensionalidad" congele Streamlit
    DIVISIONES_GRILLA = st.slider(
        "Resolución de Grilla", 
        2, 10, 5, 
        step=1, 
        help="¡Precaución! Valores altos incrementan exponencialmente el tiempo de cómputo (Máx recomendado: 10)."
    )
with col_s4:
    st.write("")
    st.write("")
    ejecutar = st.button("🔁 Ejecutar DP")

# --------------------------------------------------------------------------- #
# Generación de Grilla
# --------------------------------------------------------------------------- #
def generar_composiciones(n_activos, suma_objetivo):
    """Generador recursivo ultra-rápido que halla combinaciones enteras exactas."""
    if n_activos == 1:
        yield (suma_objetivo,)
        return
    for i in range(suma_objetivo + 1):
        for tail in generar_composiciones(n_activos - 1, suma_objetivo - i):
            yield (i,) + tail

def generar_grilla_optimizada(N, divisiones, max_cash):
    """Construye la grilla sin evaluar escenarios inútiles y respetando MAX_CASH."""
    grilla = []
    # Genera iteraciones enteras (ej: 0, 1, 2, 3... hasta 'divisiones')
    for w_int in generar_composiciones(N, divisiones):
        # Transforma los enteros a pesos porcentuales perfectos
        w_float = np.array(w_int, dtype=float) / divisiones
        
        # Filtra si el activo refugio (último) respeta el límite
        if w_float[-1] <= (max_cash + 1e-7):  
            grilla.append(w_float)
            
    return np.array(grilla)

# --------------------------------------------------------------------------- #
# Ejecución del modelo DP
# --------------------------------------------------------------------------- #
if ejecutar:
    np.random.seed(42)
    precios = descargar_precios(tickers_lista, fecha_ini, fecha_fin)
    if precios.empty or precios.shape[1] == 0:
        st.error("No se pudieron descargar datos válidos para los tickers indicados.")
        st.stop()

    tickers_validos = list(precios.columns)
    retornos = np.log(precios / precios.shift(1)).dropna()
    retornos['CASH'] = RF / DIAS_ANIO
    tickers_optimizacion = list(retornos.columns)
    N = len(tickers_optimizacion)
    mu_vec = retornos.mean().values * DIAS_ANIO
    Sigma = retornos.cov().values * DIAS_ANIO

    limites_produccion = [(0.0, 1.0)] * (N - 1) + [(0.0, MAX_CASH)]
    # Portafolio objetivo: mínima varianza
    res = minimize(lambda w: np.sqrt(w @ Sigma @ w), 
                   np.ones(N) / N, 
                   method="SLSQP", 
                   bounds=limites_produccion,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w_objetivo = res.x

    with st.spinner("Construyendo grilla de estados y evaluando Bellman..."):
        # ¡Llamamos a la nueva función optimizada!
        grilla = generar_grilla_optimizada(N, DIVISIONES_GRILLA, MAX_CASH)
        
        if len(grilla) == 0:
            st.error("⚠️ La grilla quedó vacía tras aplicar los límites de efectivo.")
            st.stop()
        
    G = len(grilla)

    # Complejidad
    operaciones = G * G * T_PERIODOS
    if operaciones > 8_000_000:
        st.error(
            f"⚠️ La combinación elegida genera {G} estados ({operaciones:,} operaciones), "
            "demasiado costosa para ejecutar en tiempo razonable. Reduce la **resolución de"
            "grilla** o reduce **T** para continuar."
        )
        st.stop()

    def costo_transaccion(w_actual, w_nuevo):
        return LAMBDA_TC * np.sum(np.abs(w_nuevo - w_actual))

    def costo_suboptimalidad(w):
        return np.sqrt((w - w_objetivo) @ Sigma @ (w - w_objetivo))

    def idx_mas_cercano(w):                                             
        # OPTIMIZACIÓN: Usar la suma de cuadrados (x^2) en lugar de np.linalg.norm
        # Esto elimina la operación de raíz cuadrada y acelera la búsqueda en la grilla un 1000%
        return int(np.argmin(np.sum((grilla - w)**2, axis=1)))

    # Retornos acumulados por periodo
    dias_por_periodo = max(1, len(retornos) // T_PERIODOS)
    retornos_periodo = []
    for t in range(T_PERIODOS):
        ini, fin = t * dias_por_periodo, min((t + 1) * dias_por_periodo, len(retornos))
        retornos_periodo.append(retornos.iloc[ini:fin].sum().values)

    # Backward induction de Bellman
    J_star = np.zeros((T_PERIODOS + 1, G))
    politica = np.full((T_PERIODOS, G), -1, dtype=int)

    barra = st.progress(0, text="Ejecutando backward induction...")
    with st.spinner("Resolviendo la ecuación de Bellman..."):
        s_next_cache = np.zeros((T_PERIODOS, G), dtype=int)
        eps_cache = np.array([costo_suboptimalidad(grilla[a]) for a in range(G)])
        for t in range(T_PERIODOS):
            for a in range(G):
                w_evol = grilla[a] * np.exp(retornos_periodo[t])
                w_evol /= w_evol.sum()
                s_next_cache[t, a] = idx_mas_cercano(w_evol)

        for t in range(T_PERIODOS - 1, -1, -1):
            for s in range(G):
                w_actual = grilla[s]
                tc = LAMBDA_TC * np.abs(grilla - w_actual).sum(axis=1)
                costo = tc + eps_cache + J_star[t + 1, s_next_cache[t]]
                mejor = int(np.argmin(costo))
                J_star[t, s] = costo[mejor]
                politica[t, s] = mejor
            barra.progress((T_PERIODOS - t) / T_PERIODOS,
                           text=f"Periodo t={t} resuelto")
    barra.empty()

    # Simulación de las 3 estrategias
    ret_simples = precios.pct_change().dropna()
    ret_simples['CASH'] = RF / DIAS_ANIO

    def simular(w_init, rebalancear_fn):
        riqueza = [capital]
        w_t = w_init.copy()
        costos, n_reb = 0.0, 0
        rebalanceo_fechas = []
        rebalanceo_periodos = []
        for i in range(len(ret_simples)):
            r = ret_simples.iloc[i].values
            riqueza.append(riqueza[-1] * (1 + w_t @ r))
            if i > 0 and i % dias_por_periodo == 0:
                periodo_actual = i // dias_por_periodo
                w_nuevo = rebalancear_fn(w_t, periodo_actual)
                if not np.allclose(w_nuevo, w_t, atol=0.01):
                    costos += costo_transaccion(w_t, w_nuevo) * riqueza[-1]
                    n_reb += 1
                    rebalanceo_fechas.append(ret_simples.index[i])
                    rebalanceo_periodos.append(periodo_actual)
                w_t = w_nuevo.copy()
            else:
                w_t = w_t * (1 + r)
                w_t /= w_t.sum()
        return riqueza, costos, n_reb, rebalanceo_fechas, rebalanceo_periodos

    def dp_rebalanceo(w_t, t_periodo):
        if t_periodo < T_PERIODOS:
            return grilla[politica[t_periodo, idx_mas_cercano(w_t)]]
        return w_t

    riq_bh, _, _, _, _ = simular(w_objetivo, lambda w, t: w)
    riq_dp, costos_dp, n_reb_dp, reb_fechas_dp, reb_periodos_dp = simular(w_objetivo, dp_rebalanceo)
    riq_sr, costos_sr, n_reb_sr, _, _ = simular(w_objetivo, lambda w, t: w_objetivo)

    # Calcular ratios de Sharpe para las estrategias
    def calcular_sharpe(riq_serie):
        s = pd.Series(riq_serie)
        rets = s.pct_change().dropna()
        if rets.std() == 0:
            return 0.0
        return (rets.mean() * DIAS_ANIO - RF) / (rets.std() * np.sqrt(DIAS_ANIO))

    sharpe_bh = calcular_sharpe(riq_bh)
    sharpe_dp = calcular_sharpe(riq_dp)
    sharpe_sr = calcular_sharpe(riq_sr)

    fechas = [precios.index[0]] + list(ret_simples.index)
    fechas_str = [str(f.date()) for f in fechas]
    reb_fechas_dp_str = [str(f.date()) for f in reb_fechas_dp]

    # Guardar en session_state
    st.session_state["dp_riq_bh"] = riq_bh
    st.session_state["dp_riq_dp"] = riq_dp
    st.session_state["dp_riq_sr"] = riq_sr
    st.session_state["dp_costos_dp"] = costos_dp
    st.session_state["dp_costos_sr"] = costos_sr
    st.session_state["dp_n_reb_dp"] = n_reb_dp
    st.session_state["dp_n_reb_sr"] = n_reb_sr
    st.session_state["dp_sharpe_bh"] = sharpe_bh
    st.session_state["dp_sharpe_dp"] = sharpe_dp
    st.session_state["dp_sharpe_sr"] = sharpe_sr
    st.session_state["dp_reb_fechas_dp_str"] = reb_fechas_dp_str
    st.session_state["dp_reb_periodos_dp"] = reb_periodos_dp
    st.session_state["dp_J_star"] = J_star
    st.session_state["dp_T_periodos"] = T_PERIODOS
    st.session_state["dp_grilla_len"] = G
    st.session_state["dp_fechas_str"] = fechas_str
    st.session_state["dp_tickers_validos"] = tickers_validos
    st.session_state["dp_w_objetivo"] = w_objetivo
    st.session_state["dp_ejecutado"] = True

    # Guardar para el módulo de Comparación
    st.session_state["dp_metricas"] = {
        "riqueza_bh": float(riq_bh[-1]),
        "riqueza_dp": float(riq_dp[-1]),
        "riqueza_sr": float(riq_sr[-1]),
        "costos_dp": float(costos_dp),
    }
    st.session_state["dp_pesos"] = dict(zip(tickers_optimizacion, w_objetivo.tolist()))

# --------------------------------------------------------------------------- #
# Renderizar UI con datos de session_state si está ejecutado
# --------------------------------------------------------------------------- #
if st.session_state.get("dp_ejecutado"):
    riq_bh = st.session_state["dp_riq_bh"]
    riq_dp = st.session_state["dp_riq_dp"]
    riq_sr = st.session_state["dp_riq_sr"]
    costos_dp = st.session_state["dp_costos_dp"]
    costos_sr = st.session_state["dp_costos_sr"]
    n_reb_dp = st.session_state["dp_n_reb_dp"]
    n_reb_sr = st.session_state["dp_n_reb_sr"]
    sharpe_bh = st.session_state["dp_sharpe_bh"]
    sharpe_dp = st.session_state["dp_sharpe_dp"]
    sharpe_sr = st.session_state["dp_sharpe_sr"]
    reb_fechas_dp_str = st.session_state["dp_reb_fechas_dp_str"]
    reb_periodos_dp = st.session_state["dp_reb_periodos_dp"]
    J_star = st.session_state["dp_J_star"]
    T_PERIODOS = st.session_state["dp_T_periodos"]
    G = st.session_state["dp_grilla_len"]
    fechas_str = st.session_state["dp_fechas_str"]
    tickers_validos = st.session_state["dp_tickers_validos"]
    w_objetivo = st.session_state["dp_w_objetivo"]

    st.success("✅ Modelo de Programación Dinámica ejecutado correctamente.")

    # Tarjetas st.metric() con riqueza final, Sharpe y costos acumulados
    st.markdown("### Métricas de las estrategias")
    c_bh, c_dp, c_sr = st.columns(3)
    
    with c_bh:
        st.markdown(f"<div style='border:1px solid #E3E6EB;border-top:4px solid {GRANATE};padding:1rem;border-radius:8px'>", unsafe_allow_html=True)
        st.subheader("Buy & Hold")
        st.metric("Riqueza Final", f"${riq_bh[-1]:,.0f}")
        st.metric("Sharpe Ratio", f"{sharpe_bh:.4f}")
        st.metric("Costos Acumulados", "$0")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_dp:
        st.markdown(f"<div style='border:1px solid #E3E6EB;border-top:4px solid {AZUL};padding:1rem;border-radius:8px'>", unsafe_allow_html=True)
        st.subheader("DP Optimizado")
        st.metric("Riqueza Final", f"${riq_dp[-1]:,.0f}")
        st.metric("Sharpe Ratio", f"{sharpe_dp:.4f}")
        st.metric("Costos Acumulados", f"${costos_dp:,.0f}", delta=f"{n_reb_dp} rebalanceos")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_sr:
        st.markdown(f"<div style='border:1px solid #E3E6EB;border-top:4px solid {VERDE};padding:1rem;border-radius:8px'>", unsafe_allow_html=True)
        st.subheader("Siempre Rebalanceado")
        st.metric("Riqueza Final", f"${riq_sr[-1]:,.0f}")
        st.metric("Sharpe Ratio", f"{sharpe_sr:.4f}")
        st.metric("Costos Acumulados", f"${costos_sr:,.0f}", delta=f"{n_reb_sr} rebalanceos")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico de evolución de riqueza — 3 curvas coloreadas + marcadores de rebalanceo
    st.markdown("#### Evolución de la riqueza ($)")
    fechas = pd.to_datetime(fechas_str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=riq_bh, mode="lines",
                             line=dict(color=GRANATE, width=2),
                             name=f"Buy & Hold (${riq_bh[-1]:,.0f})"))
    fig.add_trace(go.Scatter(x=fechas, y=riq_dp, mode="lines",
                             line=dict(color=AZUL, width=2.5, dash="dash"),
                             name=f"DP Optimizado (${riq_dp[-1]:,.0f})"))
    fig.add_trace(go.Scatter(x=fechas, y=riq_sr, mode="lines",
                             line=dict(color=VERDE, width=2, dash="dot"),
                             name=f"Siempre Rebalanceado (${riq_sr[-1]:,.0f})"))
    fig.add_hline(y=capital, line=dict(color="gray", dash="dash"), opacity=0.5)

    # Agregar marcadores en los puntos donde ocurrió rebalanceo DP
    if reb_fechas_dp_str:
        reb_fechas = pd.to_datetime(reb_fechas_dp_str)
        wealth_at_reb = [riq_dp[fechas_str.index(f)] for f in reb_fechas_dp_str]
        fig.add_trace(go.Scatter(
            x=reb_fechas, y=wealth_at_reb, mode="markers",
            marker=dict(size=12, color=DORADO, symbol="triangle-up", line=dict(color="black", width=1)),
            name="Puntos de Rebalanceo DP"
        ))

    fig.update_layout(xaxis_title="Fecha", yaxis_title="Valor USD",
                      legend=dict(x=0.01, y=0.99), height=480,
                      margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Timeline de rebalanceos (en qué periodos se rebalanceó)
    st.markdown("#### Timeline de Rebalanceos (Política DP)")
    if reb_fechas_dp_str:
        df_events = pd.DataFrame({
            "Fecha": pd.to_datetime(reb_fechas_dp_str),
            "Periodo": reb_periodos_dp,
            "Estrategia": ["Rebalanceo DP"] * len(reb_fechas_dp_str)
        })
        fig_timeline = px.scatter(
            df_events, x="Fecha", y="Estrategia", text="Periodo",
            labels={"Fecha": "Fecha", "Estrategia": ""},
            height=200
        )
        fig_timeline.update_traces(
            marker=dict(size=16, color=AZUL, symbol="circle", line=dict(color="black", width=1)),
            textposition="top center",
            textfont=dict(size=10, color="white")
        )
        fig_timeline.update_layout(
            margin=dict(t=10, b=40, l=40, r=20),
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("La política DP no requirió realizar ningún rebalanceo durante este horizonte.")

    st.markdown("---")

    # Heatmap de la tabla DP — plotly.imshow
    st.markdown("#### Heatmap de la tabla DP — Costos óptimos acumulados J*(t, s)")
    n_show = min(30, G)
    indices = np.linspace(0, G - 1, n_show, dtype=int)
    fig_hm = px.imshow(
        J_star[:T_PERIODOS, indices].T,
        labels=dict(x="Periodo t", y="Estado (índice de grilla)", color="Costo óptimo"),
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig_hm.update_layout(height=420, margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Ref.: Vaezi Jezeie et al. (2022). PLoS ONE 17(8), e0271811.")

    st.markdown("---")

    # Descarga Excel de la simulación
    df_sim = pd.DataFrame({
        "Fecha": fechas_str,
        "Buy_and_Hold": riq_bh,
        "DP_Optimizado": riq_dp,
        "Siempre_Rebalanceado": riq_sr,
    })
    df_resumen = pd.DataFrame({
        "Estrategia": ["Buy & Hold", "DP Optimizado", "Siempre Rebalanceado"],
        "Riqueza_Final": [riq_bh[-1], riq_dp[-1], riq_sr[-1]],
        "Sharpe_Ratio": [sharpe_bh, sharpe_dp, sharpe_sr],
        "Costos_Transaccion": [0.0, costos_dp, costos_sr],
        "N_Rebalanceos": [0, n_reb_dp, n_reb_sr],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, index=False, sheet_name="Resumen")
        df_sim.to_excel(writer, index=False, sheet_name="Simulacion")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Descargar simulación (Excel)",
        data=buffer,
        file_name="simulacion_dp_rebalanceo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("👆 Ajusta λ_TC, T y el paso de grilla, luego pulsa **Ejecutar DP**.")

st.markdown(
    "<div class='disclaimer' style='margin-top:1rem'>⚠️ <b>Aviso:</b> Los datos son "
    "simulaciones con fines académicos y no constituyen asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
