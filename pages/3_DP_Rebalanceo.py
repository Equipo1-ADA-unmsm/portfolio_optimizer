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

import bootstrap  # noqa: F401 — debe ir antes de cualquier import de numpy/scipy (ver bootstrap.py)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
from sidebar import renderizar_sidebar
from datos import descargar_precios, TTL_PRECIOS_SEGUNDOS
from graficos_animados import agregar_animacion_reveal

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
parametros = renderizar_sidebar(mostrar_boton_ejecutar=False, detener_si_invalido=True)
tickers_lista = parametros["tickers_lista"]
fecha_ini = parametros["fecha_ini"]
fecha_fin = parametros["fecha_fin"]
capital = parametros["capital"]
MAX_CASH = parametros["max_cash"]
forzar_recalculo = parametros["forzar_recalculo"]

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
    # "🔄 Forzar recálculo" (en el sidebar) también debe disparar una nueva
    # corrida aquí, no solo invalidar la caché — de lo contrario el usuario
    # tendría que pulsar dos botones para lograrlo.
    ejecutar = st.button("🔁 Ejecutar DP") or forzar_recalculo

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
# Parte cacheada y barata del modelo DP — descarga, retornos, Sigma y el
# portafolio objetivo (mínima varianza)
#   Antes, esto vivía en la misma función cacheada que construía la grilla Y
#   resolvía el backward induction completo, cacheada por los 8 parámetros
#   juntos (incluyendo `capital`). Eso significaba que cambiar SOLO el
#   capital a invertir —que no afecta ni al portafolio objetivo ni a la
#   tabla de Bellman, solo escala la simulación final de riqueza— igual
#   forzaba reconstruir la grilla y resolver Bellman desde cero. Separarlo
#   aquí evita ese recálculo innecesario Y permite que el backward induction
#   (la parte que sí vale la pena animar) se corra aparte, periodo a
#   periodo.
#   TTL: mismo TTL_PRECIOS_SEGUNDOS que descargar_precios() en datos.py.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=TTL_PRECIOS_SEGUNDOS)
def calcular_dp_datos(tickers, inicio, fin, max_cash):
    precios, tickers_descartados = descargar_precios(tickers, inicio, fin)

    tickers_validos = list(precios.columns)
    retornos = np.log(precios / precios.shift(1)).dropna()
    retornos['CASH'] = RF / DIAS_ANIO
    tickers_optimizacion = list(retornos.columns)
    N = len(tickers_optimizacion)
    Sigma = retornos.cov().values * DIAS_ANIO

    limites_produccion = [(0.0, 1.0)] * (N - 1) + [(0.0, max_cash)]
    # Portafolio objetivo: mínima varianza
    res = minimize(lambda w: np.sqrt(w @ Sigma @ w),
                   np.ones(N) / N,
                   method="SLSQP",
                   bounds=limites_produccion,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w_objetivo = res.x

    ret_simples = precios.pct_change().dropna()
    ret_simples['CASH'] = RF / DIAS_ANIO
    fechas_str = [str(f.date()) for f in ([precios.index[0]] + list(ret_simples.index))]

    return {
        "tickers_descartados": tickers_descartados,
        "tickers_validos": tickers_validos,
        "tickers_optimizacion": tickers_optimizacion,
        "N": N,
        "Sigma": Sigma,
        "w_objetivo": w_objetivo,
        "retornos": retornos,
        "ret_simples": ret_simples,
        "fechas_str": fechas_str,
    }


# --------------------------------------------------------------------------- #
# Backward induction de Bellman — generador NO cacheado con st.cache_data a
# propósito (un generador no se puede cachear/reanudar con st.cache_data,
# solo valores de retorno completos). Cede el control (yield) al terminar
# CADA periodo t, en el MISMO ORDEN en que el algoritmo realmente lo
# resuelve (de t=T-1 hacia t=0), para poder animar el heatmap de J*(t,s)
# rellenándose de derecha a izquierda — tal como Bellman lo resuelve de
# verdad, no de izquierda a derecha como sugeriría leerlo en pantalla. La
# caché de este resultado final se maneja a mano en session_state (ver
# "_dp_cache" más abajo), igual que en Markowitz y NSGA-II.
# --------------------------------------------------------------------------- #
def generar_bellman(Sigma, w_objetivo, grilla, s_next_cache, lambda_tc, t_periodos):
    G = len(grilla)
    J_star = np.zeros((t_periodos + 1, G))
    politica = np.full((t_periodos, G), -1, dtype=int)
    eps_cache = np.array([
        np.sqrt((grilla[a] - w_objetivo) @ Sigma @ (grilla[a] - w_objetivo))
        for a in range(G)
    ])

    for t in range(t_periodos - 1, -1, -1):
        for s in range(G):
            w_actual = grilla[s]
            tc = lambda_tc * np.abs(grilla - w_actual).sum(axis=1)
            costo = tc + eps_cache + J_star[t + 1, s_next_cache[t]]
            mejor = int(np.argmin(costo))
            J_star[t, s] = costo[mejor]
            politica[t, s] = mejor
        yield t, t_periodos, J_star.copy(), politica.copy()


def construir_fig_heatmap(J_star, t_periodos, G):
    """Arma la figura de Plotly del heatmap J*(t, s). Se usa tanto en cada
    frame de la animación (con la tabla parcialmente resuelta) como en el
    resultado final (ya completa)."""
    n_show = min(30, G)
    indices = np.linspace(0, G - 1, n_show, dtype=int)
    fig_hm = px.imshow(
        J_star[:t_periodos, indices].T,
        labels=dict(x="Periodo t", y="Estado (índice de grilla)", color="Costo óptimo"),
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig_hm.update_layout(height=420, margin=dict(t=20, b=40, l=40, r=20))
    return fig_hm


# --------------------------------------------------------------------------- #
# Ejecución del modelo DP — se dispara al pulsar "🔁 Ejecutar DP" (o
# "🔄 Forzar recálculo", que además invalida la caché antes de resolver)
# --------------------------------------------------------------------------- #
if ejecutar:
    if forzar_recalculo:
        descargar_precios.clear()
        calcular_dp_datos.clear()
        # El backward induction ya no vive en un @st.cache_data (ver nota
        # junto a generar_bellman): su caché es manual, en session_state,
        # así que "Forzar recálculo" también debe vaciarla para garantizar
        # que se vuelva a animar desde cero.
        st.session_state["_dp_cache"] = {}
        st.caption("🔄 Forzando recálculo completo (caché descartada).")

    # Guardamos SOLO los argumentos con los que se llama a calcular_dp_datos()
    # y generar_bellman(), no su resultado (series de riqueza, tabla J*
    # completa, etc.): ese resultado ya vive cacheado (uno en st.cache_data,
    # el otro a mano en session_state) bajo esta misma combinación de
    # parámetros. J_star en particular puede ser una matriz de tamaño
    # (T+1) x G nada despreciable — duplicarla en session_state por cada
    # sesión abierta era el mayor consumo de memoria evitable de los 4
    # módulos. Guardar solo esta tupla de 8 valores (en vez de ~19 claves
    # con datos pesados) reduce bastante el footprint por sesión.
    st.session_state["dp_calc_args"] = (
        tuple(tickers_lista), str(fecha_ini), str(fecha_fin), float(capital), float(MAX_CASH),
        float(LAMBDA_TC), int(T_PERIODOS), int(DIVISIONES_GRILLA),
    )
    st.session_state["dp_ejecutado"] = True

# --------------------------------------------------------------------------- #
# Renderizar UI
#   1) calcular_dp_datos() — parte barata y cacheada (descarga, Sigma,
#      portafolio objetivo). Cache-hit instantáneo si tickers/fechas/límite
#      de efectivo no cambiaron (ya no depende de capital/λ_TC/T/grilla).
#   2) El backward induction en sí: si ya se corrió antes con ESTOS MISMOS 8
#      parámetros, se reutiliza de la caché manual "_dp_cache" sin animar
#      de nuevo. Si es una corrida nueva, se anima en vivo: el heatmap
#      J*(t, s) se redibuja periodo a periodo, de derecha (t=T-1) a
#      izquierda (t=0) — el mismo orden real en que Bellman lo resuelve.
# --------------------------------------------------------------------------- #
if st.session_state.get("dp_ejecutado"):
    (tickers_calc, fecha_ini_calc, fecha_fin_calc, capital_calc, max_cash_calc,
     lambda_tc_calc, t_periodos_calc, divisiones_calc) = st.session_state["dp_calc_args"]
    T_PERIODOS = t_periodos_calc

    with st.spinner("Descargando datos y resolviendo el portafolio objetivo..."):
        datos_base = calcular_dp_datos(tickers_calc, fecha_ini_calc, fecha_fin_calc, max_cash_calc)

    tickers_descartados = datos_base["tickers_descartados"]
    tickers_validos = datos_base["tickers_validos"]
    tickers_optimizacion = datos_base["tickers_optimizacion"]
    N = datos_base["N"]
    Sigma = datos_base["Sigma"]
    w_objetivo = datos_base["w_objetivo"]
    retornos = datos_base["retornos"]
    ret_simples = datos_base["ret_simples"]
    fechas_str = datos_base["fechas_str"]

    dp_cache = st.session_state.setdefault("_dp_cache", {})
    clave_dp = st.session_state["dp_calc_args"]

    st.markdown("#### Heatmap de la tabla DP — Costos óptimos acumulados J*(t, s)")
    placeholder_hm_titulo = st.empty()
    placeholder_hm = st.empty()

    if clave_dp in dp_cache:
        # Cache-hit: ya se resolvió Bellman antes con estos mismos 8
        # parámetros. No se anima de nuevo, se muestra el resultado final.
        resultado_dp = dp_cache[clave_dp]
        placeholder_hm.plotly_chart(
            construir_fig_heatmap(resultado_dp["J_star"], T_PERIODOS, resultado_dp["G"]),
            width='stretch', key="heatmap_cacheado",
        )
    else:
        # Cache-miss: corrida genuinamente nueva.
        with st.spinner("Construyendo grilla de estados..."):
            grilla = generar_grilla_optimizada(N, divisiones_calc, max_cash_calc)
            if len(grilla) == 0:
                st.error("⚠️ La grilla quedó vacía tras aplicar los límites de efectivo.")
                st.stop()

        G = len(grilla)
        operaciones = G * G * t_periodos_calc
        if operaciones > 8_000_000:
            st.error(
                f"⚠️ La combinación elegida genera {G} estados ({operaciones:,} operaciones), "
                "demasiado costosa para ejecutar en tiempo razonable. Reduce la **resolución de "
                "grilla** o reduce **T** para continuar."
            )
            st.stop()

        def costo_transaccion(w_actual, w_nuevo):
            return lambda_tc_calc * np.sum(np.abs(w_nuevo - w_actual))

        def idx_mas_cercano(w):
            # OPTIMIZACIÓN: suma de cuadrados en vez de np.linalg.norm, para
            # evitar la raíz cuadrada en cada búsqueda sobre la grilla.
            return int(np.argmin(np.sum((grilla - w) ** 2, axis=1)))

        dias_por_periodo = max(1, len(retornos) // t_periodos_calc)
        retornos_periodo = []
        for t in range(t_periodos_calc):
            ini, fin_p = t * dias_por_periodo, min((t + 1) * dias_por_periodo, len(retornos))
            retornos_periodo.append(retornos.iloc[ini:fin_p].sum().values)

        with st.spinner("Precalculando transiciones de estado…"):
            s_next_cache = np.zeros((t_periodos_calc, G), dtype=int)
            for t in range(t_periodos_calc):
                for a in range(G):
                    w_evol = grilla[a] * np.exp(retornos_periodo[t])
                    w_evol /= w_evol.sum()
                    s_next_cache[t, a] = idx_mas_cercano(w_evol)

        # Backward induction ANIMADO: se resuelve (y se dibuja) de derecha a
        # izquierda, de t=T-1 hacia t=0 — el mismo orden real de Bellman, no
        # el orden izquierda-a-derecha que sugeriría leer la tabla.
        paso = max(1, t_periodos_calc // 30)  # ~30 actualizaciones en total
        J_star, politica = None, None
        for t, total_t, J_star_parcial, politica_parcial in generar_bellman(
            Sigma, w_objetivo, grilla, s_next_cache, lambda_tc_calc, t_periodos_calc,
        ):
            J_star, politica = J_star_parcial, politica_parcial
            if t % paso == 0 or t == 0:
                placeholder_hm_titulo.caption(
                    f"⏪ Resolviendo Bellman hacia atrás… periodo t={t} (faltan {t} de {total_t})"
                )
                placeholder_hm.plotly_chart(
                    construir_fig_heatmap(J_star, T_PERIODOS, G),
                    width='stretch', key=f"heatmap_frame_{t}",
                )
        placeholder_hm_titulo.empty()

        # Simulación de las 3 estrategias — rápida, no se anima.
        def simular(w_init, rebalancear_fn):
            riqueza = [capital_calc]
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
            if t_periodo < t_periodos_calc:
                return grilla[politica[t_periodo, idx_mas_cercano(w_t)]]
            return w_t

        riq_bh, _, _, _, _ = simular(w_objetivo, lambda w, t: w)
        riq_dp, costos_dp, n_reb_dp, reb_fechas_dp, reb_periodos_dp = simular(w_objetivo, dp_rebalanceo)
        riq_sr, costos_sr, n_reb_sr, _, _ = simular(w_objetivo, lambda w, t: w_objetivo)

        def calcular_sharpe(riq_serie):
            s = pd.Series(riq_serie)
            rets = s.pct_change().dropna()
            if rets.std() == 0:
                return 0.0
            return (rets.mean() * DIAS_ANIO - RF) / (rets.std() * np.sqrt(DIAS_ANIO))

        sharpe_bh = calcular_sharpe(riq_bh)
        sharpe_dp = calcular_sharpe(riq_dp)
        sharpe_sr = calcular_sharpe(riq_sr)
        reb_fechas_dp_str = [str(f.date()) for f in reb_fechas_dp]

        resultado_dp = {
            "riq_bh": riq_bh, "riq_dp": riq_dp, "riq_sr": riq_sr,
            "costos_dp": costos_dp, "costos_sr": costos_sr,
            "n_reb_dp": n_reb_dp, "n_reb_sr": n_reb_sr,
            "sharpe_bh": sharpe_bh, "sharpe_dp": sharpe_dp, "sharpe_sr": sharpe_sr,
            "reb_fechas_dp_str": reb_fechas_dp_str, "reb_periodos_dp": reb_periodos_dp,
            "J_star": J_star, "G": G,
        }
        dp_cache[clave_dp] = resultado_dp

        # Redibuja la versión final (el último frame animado puede quedar un
        # par de periodos antes de t=0 según el paso de la animación).
        placeholder_hm.plotly_chart(
            construir_fig_heatmap(J_star, T_PERIODOS, G),
            width='stretch', key="heatmap_final",
        )

    riq_bh = resultado_dp["riq_bh"]
    riq_dp = resultado_dp["riq_dp"]
    riq_sr = resultado_dp["riq_sr"]
    costos_dp = resultado_dp["costos_dp"]
    costos_sr = resultado_dp["costos_sr"]
    n_reb_dp = resultado_dp["n_reb_dp"]
    n_reb_sr = resultado_dp["n_reb_sr"]
    sharpe_bh = resultado_dp["sharpe_bh"]
    sharpe_dp = resultado_dp["sharpe_dp"]
    sharpe_sr = resultado_dp["sharpe_sr"]
    reb_fechas_dp_str = resultado_dp["reb_fechas_dp_str"]
    reb_periodos_dp = resultado_dp["reb_periodos_dp"]
    J_star = resultado_dp["J_star"]
    G = resultado_dp["G"]

    st.caption("Ref.: Vaezi Jezeie et al. (2022). PLoS ONE 17(8), e0271811.")
    st.markdown("---")

    # Guardar para el módulo de Comparación (pequeño: dicts de floats, no la
    # tabla J* ni las series de riqueza). Se usan los parámetros REALMENTE
    # calculados (tickers_calc, etc.), no los valores actuales del sidebar,
    # que podrían haber cambiado desde la última corrida sin que el usuario
    # vuelva a pulsar "Ejecutar DP".
    st.session_state["dp_metricas"] = {
        "riqueza_bh": float(riq_bh[-1]),
        "riqueza_dp": float(riq_dp[-1]),
        "riqueza_sr": float(riq_sr[-1]),
        "costos_dp": float(costos_dp),
    }
    st.session_state["dp_pesos"] = dict(zip(tickers_optimizacion, w_objetivo.tolist()))
    st.session_state["dp_params"] = (
        tickers_calc, fecha_ini_calc, fecha_fin_calc, capital_calc, max_cash_calc,
    )

    if tickers_descartados:
        st.warning(
            "⚠️ Se descartaron los siguientes tickers por no tener datos válidos en "
            f"Yahoo Finance para el rango de fechas seleccionado: {', '.join(tickers_descartados)}. "
            "El análisis continuó con el resto del universo."
        )

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
                      margin=dict(t=60, b=40, l=40, r=20))
    fig = agregar_animacion_reveal(fig)
    st.plotly_chart(fig, width='stretch')

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
            margin=dict(t=60, b=40, l=40, r=20),
            yaxis=dict(showticklabels=False)
        )
        fig_timeline = agregar_animacion_reveal(fig_timeline)
        st.plotly_chart(fig_timeline, width='stretch')
    else:
        st.info("La política DP no requirió realizar ningún rebalanceo durante este horizonte.")

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
