"""
Módulo 1 — Datos y Markowitz
============================
Descarga datos de Yahoo Finance, calcula retornos logarítmicos, resuelve el
portafolio de Markowitz (máx Sharpe y mín varianza con scipy.optimize), genera
la frontera eficiente con 200 puntos y simula la evolución de riqueza.

Los parámetros (tickers, fechas, capital) provienen de st.session_state,
configurados en el sidebar del homepage (app.py).
"""

import io
import warnings

import bootstrap  # noqa: F401 — debe ir antes de cualquier import de numpy/scipy (ver bootstrap.py)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
from sidebar import renderizar_sidebar
from datos import descargar_precios, TTL_PRECIOS_SEGUNDOS
from finanzas import (
    portfolio_performance,
    negative_sharpe_ratio,
    portfolio_volatility,
    calculate_sortino_ratio,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuración de página y paleta
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Datos y Markowitz", page_icon="📊", layout="wide")

# Estilos (paleta, tipografía, ajuste de modo oscuro y renombrado del sidebar)
#   Definidos en estilos.py para reutilizarse igual en todos los módulos.
aplicar_estilos()

DIAS_ANIO = 252
RF_RATE = 0.02  # Retorno anualizado del activo CASH (2%)

st.markdown(
    "<h1>📊 Módulo 1 · Datos y Markowitz</h1>",
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

# --------------------------------------------------------------------------- #
# Parámetros desde session_state (con fallback a defaults)
# --------------------------------------------------------------------------- #
TICKERS = st.session_state.get("tickers", ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"])
START_DATE = str(st.session_state.get("fecha_ini", "2015-01-01"))
END_DATE = str(st.session_state.get("fecha_fin", "2024-12-31"))
CAPITAL_INICIAL = float(st.session_state.get("capital", 100_000))
MAX_CASH_LIMIT = float(st.session_state.get("max_cash", 0.20))

st.caption(
    f"**Universo:** {', '.join(TICKERS)}  |  **Periodo:** {START_DATE} → {END_DATE}  "
    f"|  **Capital:** ${CAPITAL_INICIAL:,.0f}  |  🛡️ **Límite Efectivo:** {MAX_CASH_LIMIT:.0%}"
)

if not TICKERS:
    # Salvaguarda extra: en teoría ya no debería llegar aquí, porque
    # renderizar_sidebar(detener_si_invalido=True) corta la ejecución antes
    # si no hay tickers válidos. Se deja por robustez ante ediciones futuras.
    st.error("⚠️ No hay tickers configurados. Vuelve al inicio y define el universo.")
    st.stop()


# --------------------------------------------------------------------------- #
# Cálculo pesado de Markowitz — cacheado con st.cache_data
#   Antes, este bloque se recalculaba por completo cada vez que se pulsaba
#   "🚀 Ejecutar Análisis", incluso si los parámetros (tickers, fechas,
#   capital, límite de efectivo) no habían cambiado desde la última vez
#   (p. ej. el usuario navega a otra página y vuelve, y pulsa el botón de
#   nuevo). Eso repetía 202 llamadas a scipy.optimize.minimize (1 máximo
#   Sharpe + 1 mínima varianza + 200 puntos de la frontera eficiente) de
#   forma innecesaria. Al cachear por (tickers, inicio, fin, capital,
#   max_cash_limit), una repetición con los mismos parámetros se sirve
#   instantáneamente desde caché.
#   TTL: se le da el mismo TTL_PRECIOS_SEGUNDOS que usa descargar_precios()
#   en datos.py. Sin esto, un resultado de Markowitz podría quedar cacheado
#   indefinidamente aunque los precios de mercado ya se hayan refrescado
#   (TTL de precios cumplido) — el usuario vería un resultado "viejo" que
#   ya no corresponde a datos igual de recientes. Con el mismo TTL en
#   ambas capas, cuando los precios se refrescan, el cálculo de Markowitz
#   que depende de ellos también se vuelve a resolver.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=TTL_PRECIOS_SEGUNDOS)
def calcular_markowitz(tickers, inicio, fin, capital_inicial, max_cash_limit):
    np.random.seed(42)

    # 1. Datos y retornos
    df_prices, tickers_descartados = descargar_precios(tickers, inicio, fin)

    tickers_validos = list(df_prices.columns)
    log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

    mu = log_returns.mean() * DIAS_ANIO
    Sigma = log_returns.cov() * DIAS_ANIO

    # 2. Inclusión de activo CASH (risk-free)
    TICKERS_EXT = tickers_validos + ["CASH"]
    mu["CASH"] = RF_RATE
    Sigma.loc["CASH"] = 0.0
    Sigma["CASH"] = 0.0
    log_returns["CASH"] = RF_RATE / DIAS_ANIO

    num_assets = len(TICKERS_EXT)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    limites_produccion = [(0.0, 1.0)] * (num_assets - 1) + [(0.0, max_cash_limit)]
    bounds = tuple(limites_produccion)
    init_guess = np.array(num_assets * [1.0 / num_assets])

    # Si el guess inicial supera el límite de efectivo, lo rebalanceamos para no causar errores en SLSQP
    if init_guess[-1] > max_cash_limit:
        exceso = init_guess[-1] - max_cash_limit
        init_guess[-1] = max_cash_limit
        init_guess[:-1] += exceso / (num_assets - 1)

    # 3. Máximo Sharpe
    opt_sharpe = minimize(
        negative_sharpe_ratio, init_guess, args=(mu, Sigma, RF_RATE),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    pesos_sharpe = opt_sharpe.x
    ret_sharpe, vol_sharpe = portfolio_performance(pesos_sharpe, mu, Sigma)
    ratio_sharpe_opt = (ret_sharpe - RF_RATE) / vol_sharpe
    ratio_sortino_opt = calculate_sortino_ratio(pesos_sharpe, log_returns, RF_RATE)

    # 4. Mínima Varianza
    opt_minvar = minimize(
        portfolio_volatility, init_guess, args=(mu, Sigma),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    pesos_minvar = opt_minvar.x
    ret_minvar, vol_minvar = portfolio_performance(pesos_minvar, mu, Sigma)

    # NOTA: la frontera eficiente (200 optimizaciones) YA NO se calcula aquí
    # adentro. Antes vivía en este mismo bucle, pero eso significaba que la
    # página solo podía mostrar los 200 puntos ya terminados, todos de golpe,
    # sin forma de animar su construcción mientras scipy.optimize los resuelve
    # uno a uno. Se movió a `generar_frontera_eficiente()` (generador, más
    # abajo), que la página consume punto a punto para poder redibujar el
    # gráfico en vivo. Aquí solo se devuelven `bounds` e `init_guess`, que es
    # todo lo que ese generador necesita además de mu/Sigma (ya en el dict).

    return {
        "tickers_validos": tickers_validos,
        "tickers_descartados": tickers_descartados,
        "tickers_ext": TICKERS_EXT,
        "mu": mu,
        "sigma": Sigma,
        "log_returns": log_returns,
        "pesos_sharpe": pesos_sharpe,
        "ret_sharpe": ret_sharpe,
        "vol_sharpe": vol_sharpe,
        "ratio_sharpe": ratio_sharpe_opt,
        "ratio_sortino": ratio_sortino_opt,
        "pesos_minvar": pesos_minvar,
        "ret_minvar": ret_minvar,
        "vol_minvar": vol_minvar,
        "bounds": bounds,
        "init_guess": init_guess,
    }


# --------------------------------------------------------------------------- #
# Generador de la frontera eficiente — NO cacheado con st.cache_data a
# propósito: es un generador (yield), y st.cache_data no sabe cachear ni
# reanudar generadores, solo valores de retorno completos. La caché de ESTE
# resultado se maneja a mano en session_state (ver "_frontera_cache" más
# abajo), justo para poder distinguir un cache-hit (no animar, solo mostrar
# el resultado ya calculado) de un cache-miss (sí animar, punto por punto,
# mientras scipy.optimize realmente los va resolviendo).
# --------------------------------------------------------------------------- #
def generar_frontera_eficiente(mu, Sigma, bounds, init_guess, n_puntos=200):
    """Resuelve el problema de mínima varianza para cada retorno objetivo de
    la frontera eficiente, cediendo el control después de cada optimización
    individual (yield) en vez de devolver los 200 puntos ya calculados.

    Yields
    ------
    (i, n_puntos, vol, ret, exito) : tupla con el progreso de la iteración i
        de n_puntos totales. `vol`/`ret` son None si esa optimización en
        particular no convergió (`exito=False`).
    """
    target_returns = np.linspace(mu.min(), mu.max(), n_puntos)
    for i, target in enumerate(target_returns):
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, mu, Sigma)[0] - t},
        )
        res = minimize(
            portfolio_volatility, init_guess, args=(mu, Sigma),
            method="SLSQP", bounds=bounds, constraints=cons,
        )
        if res.success:
            yield i, n_puntos, res.fun, target, True
        else:
            yield i, n_puntos, None, None, False


def construir_fig_frontera(efficient_vols, efficient_rets, vols_activos, rets_activos,
                            nombres_activos, vol_sharpe, ret_sharpe, vol_minvar, ret_minvar):
    """Arma la figura de Plotly de la frontera eficiente. Extraído a función
    para poder llamarla tanto en cada frame de la animación como en el
    resultado final, sin duplicar la definición de las 5 trazas."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=efficient_vols, y=efficient_rets, mode="lines+markers",
        line=dict(color=AZUL, width=1.5),
        marker=dict(color=AZUL, size=4),
        name="Frontera Eficiente",
        hovertemplate="σ: %{x:.2%}<br>E(R): %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0.0], y=[RF_RATE], mode="markers",
        marker=dict(symbol="x", color="black", size=14, line=dict(width=2)),
        name="CASH (Risk-Free)",
        hovertemplate=f"CASH<br>σ: 0.00%<br>E(R): {RF_RATE:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=vols_activos, y=rets_activos, mode="markers+text",
        marker=dict(symbol="square", color="gray", size=10,
                    line=dict(color="black", width=1)),
        text=nombres_activos, textposition="top center",
        textfont=dict(size=10),
        name="Activos individuales",
        hovertemplate="%{text}<br>σ: %{x:.2%}<br>E(R): %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[vol_sharpe], y=[ret_sharpe], mode="markers",
        marker=dict(symbol="star", color=GRANATE, size=22,
                    line=dict(color="black", width=1)),
        name="Máximo Sharpe",
        hovertemplate="Máximo Sharpe<br>σ: %{x:.2%}<br>E(R): %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[vol_minvar], y=[ret_minvar], mode="markers",
        marker=dict(symbol="diamond", color=DORADO, size=16,
                    line=dict(color="black", width=1)),
        name="Mínima Varianza",
        hovertemplate="Mínima Varianza<br>σ: %{x:.2%}<br>E(R): %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Volatilidad Anualizada (Riesgo)",
        yaxis_title="Retorno Esperado Anualizado",
        xaxis=dict(tickformat=".1%", showgrid=True,
                   gridcolor="rgba(128,128,128,0.3)", griddash="dash"),
        yaxis=dict(tickformat=".1%", showgrid=True,
                   gridcolor="rgba(128,128,128,0.3)", griddash="dash"),
        legend=dict(x=0.01, y=0.99),
        height=520,
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


# --------------------------------------------------------------------------- #
# Disparo del cálculo — SOLO se ejecuta al pulsar "🚀 Ejecutar Análisis" (o
# "🔄 Forzar recálculo", que además invalida la caché antes de calcular)
# --------------------------------------------------------------------------- #
if ejecutar:
    if forzar_recalculo:
        descargar_precios.clear()
        calcular_markowitz.clear()
        # La frontera eficiente no vive en un @st.cache_data (ver nota junto
        # a generar_frontera_eficiente): su caché es manual, en
        # session_state, así que "Forzar recálculo" también debe vaciarla
        # por completo para garantizar que se vuelva a animar desde cero.
        st.session_state["_frontera_cache"] = {}
        st.caption("🔄 Forzando recálculo completo (caché descartada).")

    # Guardamos SOLO los argumentos con los que se llama a calcular_markowitz(),
    # no su resultado (mu, Sigma, log_returns, frontera de 200 puntos, etc.):
    # ese resultado ya vive cacheado por st.cache_data bajo esta misma
    # combinación de parámetros. Antes se copiaba también aquí, duplicando en
    # session_state — por cada sesión abierta — varios arrays y DataFrames
    # que ya estaban en la caché de la función. Guardar solo esta tupla de 5
    # valores (en vez de ~15 claves con datos pesados) reduce bastante el
    # footprint de memoria por sesión, algo que importa si la app se usa con
    # muchos usuarios concurrentes.
    st.session_state["markowitz_params"] = (
        tuple(TICKERS), START_DATE, END_DATE, CAPITAL_INICIAL, MAX_CASH_LIMIT,
    )
    st.session_state["markowitz_ejecutado"] = True

# --------------------------------------------------------------------------- #
# Renderizar resultados — reconsulta calcular_markowitz() con los parámetros
# guardados. Si ya se calculó antes con esos mismos parámetros (lo normal en
# un rerun por cambiar otro widget sin volver a pulsar el botón), esto es un
# cache-hit instantáneo: no hay descarga ni optimización de por medio.
# --------------------------------------------------------------------------- #
if st.session_state.get("markowitz_ejecutado"):
    params_calc = st.session_state["markowitz_params"]
    with st.spinner("Optimizando portafolio..."):
        resultado_mk = calcular_markowitz(*params_calc)

    tickers_validos = resultado_mk["tickers_validos"]
    tickers_descartados = resultado_mk["tickers_descartados"]
    TICKERS_EXT = resultado_mk["tickers_ext"]
    mu = resultado_mk["mu"]
    Sigma = resultado_mk["sigma"]
    log_returns = resultado_mk["log_returns"]
    pesos_sharpe = resultado_mk["pesos_sharpe"]
    ret_sharpe = resultado_mk["ret_sharpe"]
    vol_sharpe = resultado_mk["vol_sharpe"]
    ratio_sharpe_opt = resultado_mk["ratio_sharpe"]
    ratio_sortino_opt = resultado_mk["ratio_sortino"]
    pesos_minvar = resultado_mk["pesos_minvar"]
    ret_minvar = resultado_mk["ret_minvar"]
    vol_minvar = resultado_mk["vol_minvar"]
    bounds = resultado_mk["bounds"]
    init_guess = resultado_mk["init_guess"]
    CAPITAL_INICIAL = params_calc[3]

    # Guardar para el módulo de Comparación (esto sí es pequeño: un par de
    # dicts con floats, no arrays ni DataFrames — se recalcula barato en
    # cada rerun, no hace falta evitar duplicarlo).
    st.session_state["markowitz_pesos"] = dict(zip(TICKERS_EXT, pesos_sharpe))
    st.session_state["markowitz_metricas"] = {
        "retorno": ret_sharpe, "volatilidad": vol_sharpe,
        "sharpe": ratio_sharpe_opt, "sortino": ratio_sortino_opt,
    }

    if tickers_descartados:
        st.warning(
            "⚠️ Se descartaron los siguientes tickers por no tener datos válidos en "
            f"Yahoo Finance para el rango de fechas seleccionado: {', '.join(tickers_descartados)}. "
            "El análisis continuó con el resto del universo."
        )

    # ----------------------------------------------------------------------- #
    # Métricas clave — 3 columnas
    # ----------------------------------------------------------------------- #
    st.markdown("### Métricas del portafolio de máximo Sharpe")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio", f"{ratio_sharpe_opt:.4f}")
    c2.metric("Sortino Ratio", f"{ratio_sortino_opt:.4f}")
    c3.metric("Volatilidad anual", f"{vol_sharpe:.2%}")

    st.markdown("---")

    # ----------------------------------------------------------------------- #
    # Frontera eficiente (plotly) + Pie chart (plotly)
    #   Antes la frontera se dibujaba con Matplotlib (st.pyplot): era la
    #   única gráfica de las 4 páginas sin interactividad (sin hover) y la
    #   única con fondo/texto fijos que no se adaptaban al modo oscuro del
    #   navegador (a diferencia del resto del sitio, que usa Plotly). Migrada
    #   aquí para que también tenga hover con el valor exacto de cada punto y
    #   sea consistente con el resto de gráficas de la app.
    # ----------------------------------------------------------------------- #
    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        # Volatilidad/retorno individual de cada activo (excluyendo CASH),
        # para los marcadores grises con su ticker como etiqueta. Se conocen
        # de antemano (no dependen de la frontera), así que ya se pueden
        # mostrar desde el primer frame de la animación.
        vols_activos, rets_activos, nombres_activos = [], [], []
        for i, ticker in enumerate(TICKERS_EXT):
            if ticker != "CASH":
                vols_activos.append(float(np.sqrt(Sigma.iloc[i, i])))
                rets_activos.append(float(mu.iloc[i]))
                nombres_activos.append(ticker)

        placeholder_titulo = st.empty()
        placeholder_chart = st.empty()

        # ------------------------------------------------------------------- #
        # Caché manual de la frontera (independiente de calcular_markowitz).
        #   Se guarda bajo la MISMA tupla de parámetros (params_calc) que ya
        #   identifica de forma única a esta corrida. Si ya está en caché,
        #   fue calculada antes (p. ej. el usuario navegó a otra página y
        #   volvió, o cambió un widget que no afecta este cálculo) y se
        #   muestra de una sola vez, SIN animar de nuevo. Si no está, es un
        #   cálculo genuinamente nuevo: se anima en vivo mientras
        #   scipy.optimize resuelve cada punto.
        # ------------------------------------------------------------------- #
        frontera_cache = st.session_state.setdefault("_frontera_cache", {})

        if params_calc in frontera_cache:
            efficient_vols, efficient_rets = frontera_cache[params_calc]
            placeholder_chart.plotly_chart(
                construir_fig_frontera(efficient_vols, efficient_rets, vols_activos,
                                       rets_activos, nombres_activos, vol_sharpe,
                                       ret_sharpe, vol_minvar, ret_minvar),
                width='stretch', key="frontera_cacheada",
            )
        else:
            efficient_vols, efficient_rets = [], []
            n_puntos = 200
            # Redibujar en CADA una de las 200 optimizaciones sería lento
            # (serializar y enviar una figura de Plotly completa por cada
            # punto) y se vería entrecortado. Se actualiza cada `paso`
            # iteraciones (~40 actualizaciones en total) para que la
            # animación se sienta fluida sin sacrificar la fluidez del
            # navegador ni la del resto de la app.
            paso = max(1, n_puntos // 40)
            for i, total, vol, ret, exito in generar_frontera_eficiente(mu, Sigma, bounds, init_guess, n_puntos):
                if exito:
                    efficient_vols.append(vol)
                    efficient_rets.append(ret)
                if i % paso == 0 or i == total - 1:
                    placeholder_titulo.markdown(
                        f"#### Frontera Eficiente Analítica (calculando… {i + 1}/{total})"
                    )
                    placeholder_chart.plotly_chart(
                        construir_fig_frontera(efficient_vols, efficient_rets, vols_activos,
                                               rets_activos, nombres_activos, vol_sharpe,
                                               ret_sharpe, vol_minvar, ret_minvar),
                        width='stretch', key=f"frontera_frame_{i}",
                    )
            # Cálculo terminado: se guarda en la caché manual para que la
            # próxima vez (mismos parámetros) no se vuelva a animar.
            frontera_cache[params_calc] = (efficient_vols, efficient_rets)

        # El título final muestra el conteo REAL de puntos graficados, no un
        # "200" fijo: se intentan 200 optimizaciones (una por cada retorno
        # objetivo en target_returns), pero scipy.optimize.minimize puede no
        # converger para algunas de ellas (típicamente cerca de los extremos
        # de la frontera, en combinación con el límite de efectivo
        # MAX_CASH_LIMIT). Esos puntos fallidos se descartan silenciosamente
        # (`if exito: ...` arriba), así que el título debe reflejar cuántos
        # sobrevivieron, no cuántos se intentaron.
        n_puntos_frontera = len(efficient_vols)
        if n_puntos_frontera < 200:
            placeholder_titulo.markdown(f"#### Frontera Eficiente Analítica ({n_puntos_frontera}/200 puntos)")
            st.caption(
                f"⚠️ {200 - n_puntos_frontera} de los 200 retornos objetivo no convergieron "
                "(scipy.optimize) y se omitieron de la frontera, probablemente por el límite "
                "de efectivo o por acercarse a los extremos de retorno alcanzables."
            )
        else:
            placeholder_titulo.markdown(f"#### Frontera Eficiente Analítica ({n_puntos_frontera} puntos)")

    with col_der:
        st.markdown("#### Composición del portafolio (máx Sharpe)")
        df_pesos = pd.DataFrame({"Activo": TICKERS_EXT, "Peso": pesos_sharpe})
        df_pesos = df_pesos[df_pesos["Peso"] > 1e-4].sort_values("Peso", ascending=False)
        fig_pie = px.pie(
            df_pesos, names="Activo", values="Peso", hole=0.4,
            color_discrete_sequence=[AZUL, GRANATE, DORADO, "#4472C4", "#A6A6A6", "#2E7D32"],
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown("---")

    # ----------------------------------------------------------------------- #
    # Evolución de riqueza: Buy&Hold (igual ponderado) vs. Máximo Sharpe
    # ----------------------------------------------------------------------- #
    st.markdown("#### Evolución de la riqueza ($)")
    n_real = len(tickers_validos)
    w_equal = np.array([1 / n_real] * n_real + [0.0])
    ret_sharpe_daily = log_returns.dot(pesos_sharpe)
    ret_equal_daily = log_returns.dot(w_equal)
    wealth_sharpe = CAPITAL_INICIAL * np.exp(ret_sharpe_daily.cumsum())
    wealth_equal = CAPITAL_INICIAL * np.exp(ret_equal_daily.cumsum())

    df_wealth = pd.DataFrame({
        "Máximo Sharpe": wealth_sharpe,
        "Buy & Hold (igual ponderado)": wealth_equal,
    })
    st.line_chart(df_wealth)

    cf1, cf2 = st.columns(2)
    cf1.metric("Valor final · Máximo Sharpe", f"${wealth_sharpe.iloc[-1]:,.0f}")
    cf2.metric("Valor final · Buy & Hold", f"${wealth_equal.iloc[-1]:,.0f}")

    st.markdown("---")

    # ----------------------------------------------------------------------- #
    # Tabla de resultados + descarga a Excel
    # ----------------------------------------------------------------------- #
    st.markdown("#### Pesos óptimos por activo")
    sharpe_ind, vol_ind = [], []
    for t in TICKERS_EXT:
        v = np.sqrt(Sigma.loc[t, t])
        vol_ind.append(v)
        sharpe_ind.append((mu[t] - RF_RATE) / v if v > 0 else 0.0)

    df_resultados = pd.DataFrame({
        "Ticker": TICKERS_EXT,
        "Peso_Optimo": pesos_sharpe,
        "Retorno_Esperado": mu.values,
        "Volatilidad_Anual": vol_ind,
        "Sharpe_Ratio": sharpe_ind,
    }).sort_values("Peso_Optimo", ascending=False)

    st.dataframe(
        df_resultados.style.format({
            "Peso_Optimo": "{:.4%}",
            "Retorno_Esperado": "{:.4%}",
            "Volatilidad_Anual": "{:.4%}",
            "Sharpe_Ratio": "{:.4f}",
        }),
        width='stretch',
    )

    # Exportar a Excel en memoria
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_resultados.to_excel(writer, index=False, sheet_name="Pesos_Optimos")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Descargar pesos óptimos (Excel)",
        data=buffer,
        file_name="pesos_optimos_produccion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("👆 Configura los parámetros en la barra lateral y pulsa **🚀 Ejecutar Análisis** "
            "para calcular el portafolio de Markowitz.")

st.markdown(
    "<div class='disclaimer'>⚠️ <b>Aviso:</b> Los datos son simulaciones con fines "
    "académicos y no constituyen asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
