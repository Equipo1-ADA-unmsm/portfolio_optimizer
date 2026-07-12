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

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.express as px
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
from sidebar import renderizar_sidebar

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
parametros = renderizar_sidebar()
tickers_lista = parametros["tickers_lista"]
fecha_ini = parametros["fecha_ini"]
fecha_fin = parametros["fecha_fin"]
capital = parametros["capital"]
MAX_CASH = parametros["max_cash"]
ejecutar = parametros["ejecutar"]

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
    f"|  **Capital:** ${CAPITAL_INICIAL:,.0f}  |  **RF (CASH):** {MAX_CASH_LIMIT:.0%}"
)

if not TICKERS:
    st.error("⚠️ No hay tickers configurados. Vuelve al inicio y define el universo.")
    st.stop()

# --------------------------------------------------------------------------- #
# Descarga de datos (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def descargar_precios(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(axis=1, how="all").ffill().dropna()
    return df


# --------------------------------------------------------------------------- #
# Funciones de portafolio
# --------------------------------------------------------------------------- #
def portfolio_performance(weights, mu, Sigma):
    p_ret = np.sum(mu * weights)
    # NUEVO: abs() protege contra errores de dominio matemático (raíz de negativo) por redondeos
    p_std = np.sqrt(abs(np.dot(weights.T, np.dot(Sigma, weights))))
    return p_ret, p_std


def negative_sharpe_ratio(weights, mu, Sigma, risk_free_rate=0.0):
    p_ret, p_std = portfolio_performance(weights, mu, Sigma)
    return -(p_ret - risk_free_rate) / p_std if p_std > 0 else 0.0


def portfolio_volatility(weights, mu, Sigma):
    return portfolio_performance(weights, mu, Sigma)[1]


def calculate_sortino_ratio(weights, historical_returns, risk_free_rate=0.0):
    port_returns = historical_returns.dot(weights)
    downside = port_returns[port_returns < 0]
    expected_return = port_returns.mean() * DIAS_ANIO
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(DIAS_ANIO)
    if downside_std == 0:
        return 0.0
    return (expected_return - risk_free_rate) / downside_std


# --------------------------------------------------------------------------- #
# Cálculo principal — SOLO se ejecuta al pulsar "🚀 Ejecutar Análisis"
# --------------------------------------------------------------------------- #
if ejecutar:
    with st.spinner("Optimizando portafolio..."):
        np.random.seed(42)

        # 1. Datos y retornos
        df_prices = descargar_precios(TICKERS, START_DATE, END_DATE)
        if df_prices.empty or df_prices.shape[1] == 0:
            st.error("No se pudieron descargar datos válidos para los tickers indicados.")
            st.stop()

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
        limites_produccion = [(0.0, 1.0)] * (num_assets - 1) + [(0.0, MAX_CASH_LIMIT)]
        bounds = tuple(limites_produccion)
        init_guess = np.array(num_assets * [1.0 / num_assets])

        # Si el guess inicial supera el límite de efectivo, lo rebalanceamos para no causar errores en SLSQP
        if init_guess[-1] > MAX_CASH_LIMIT:
            exceso = init_guess[-1] - MAX_CASH_LIMIT
            init_guess[-1] = MAX_CASH_LIMIT
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

        # 5. Frontera eficiente (200 puntos)
        target_returns = np.linspace(mu.min(), mu.max(), 200)
        efficient_vols, efficient_rets = [], []
        for target in target_returns:
            cons = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, mu, Sigma)[0] - t},
            )
            res = minimize(
                portfolio_volatility, init_guess, args=(mu, Sigma),
                method="SLSQP", bounds=bounds, constraints=cons,
            )
            if res.success:
                efficient_vols.append(res.fun)
                efficient_rets.append(target)

    # Guardar resultados en session_state para que sobrevivan a futuros reruns
    # (por ejemplo, si el usuario cambia otro widget sin volver a pulsar el botón)
    st.session_state["mk_tickers_validos"] = tickers_validos
    st.session_state["mk_tickers_ext"] = TICKERS_EXT
    st.session_state["mk_mu"] = mu
    st.session_state["mk_sigma"] = Sigma
    st.session_state["mk_log_returns"] = log_returns
    st.session_state["mk_pesos_sharpe"] = pesos_sharpe
    st.session_state["mk_ret_sharpe"] = ret_sharpe
    st.session_state["mk_vol_sharpe"] = vol_sharpe
    st.session_state["mk_ratio_sharpe"] = ratio_sharpe_opt
    st.session_state["mk_ratio_sortino"] = ratio_sortino_opt
    st.session_state["mk_pesos_minvar"] = pesos_minvar
    st.session_state["mk_ret_minvar"] = ret_minvar
    st.session_state["mk_vol_minvar"] = vol_minvar
    st.session_state["mk_efficient_vols"] = efficient_vols
    st.session_state["mk_efficient_rets"] = efficient_rets
    st.session_state["mk_capital_inicial"] = CAPITAL_INICIAL
    st.session_state["markowitz_ejecutado"] = True

    # Guardar para el módulo de Comparación
    st.session_state["markowitz_pesos"] = dict(zip(TICKERS_EXT, pesos_sharpe))
    st.session_state["markowitz_metricas"] = {
        "retorno": ret_sharpe, "volatilidad": vol_sharpe,
        "sharpe": ratio_sharpe_opt, "sortino": ratio_sortino_opt,
    }

# --------------------------------------------------------------------------- #
# Renderizar resultados con datos de session_state, si ya se ejecutó el análisis
# --------------------------------------------------------------------------- #
if st.session_state.get("markowitz_ejecutado"):
    tickers_validos = st.session_state["mk_tickers_validos"]
    TICKERS_EXT = st.session_state["mk_tickers_ext"]
    mu = st.session_state["mk_mu"]
    Sigma = st.session_state["mk_sigma"]
    log_returns = st.session_state["mk_log_returns"]
    pesos_sharpe = st.session_state["mk_pesos_sharpe"]
    ret_sharpe = st.session_state["mk_ret_sharpe"]
    vol_sharpe = st.session_state["mk_vol_sharpe"]
    ratio_sharpe_opt = st.session_state["mk_ratio_sharpe"]
    ratio_sortino_opt = st.session_state["mk_ratio_sortino"]
    pesos_minvar = st.session_state["mk_pesos_minvar"]
    ret_minvar = st.session_state["mk_ret_minvar"]
    vol_minvar = st.session_state["mk_vol_minvar"]
    efficient_vols = st.session_state["mk_efficient_vols"]
    efficient_rets = st.session_state["mk_efficient_rets"]
    CAPITAL_INICIAL = st.session_state["mk_capital_inicial"]

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
    # Frontera eficiente (matplotlib -> st.pyplot) + Pie chart (plotly)
    # ----------------------------------------------------------------------- #
    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        st.markdown("#### Frontera Eficiente Analítica (200 puntos)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(efficient_vols, efficient_rets, color=AZUL, lw=2.5,
                label="Frontera Eficiente", zorder=2)
        ax.scatter(0.0, RF_RATE, color="black", marker="X", s=150,
                   label="CASH (Risk-Free)", zorder=4)
        for i, ticker in enumerate(TICKERS_EXT):
            if ticker != "CASH":
                v = np.sqrt(Sigma.iloc[i, i])
                r = mu.iloc[i]
                ax.scatter(v, r, marker="s", color="gray", s=60, zorder=3)
                ax.text(v + 0.003, r, ticker, fontsize=9, zorder=3)
        ax.scatter(vol_sharpe, ret_sharpe, color=GRANATE, marker="*", s=350,
                   edgecolor="black", label="Máximo Sharpe", zorder=5)
        ax.scatter(vol_minvar, ret_minvar, color=DORADO, marker="D", s=150,
                   edgecolor="black", label="Mínima Varianza", zorder=5)
        ax.set_xlabel("Volatilidad Anualizada (Riesgo)")
        ax.set_ylabel("Retorno Esperado Anualizado")
        ax.margins(0.05)
        ax.legend(loc="best", framealpha=0.9, edgecolor="black")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

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
        st.plotly_chart(fig_pie, use_container_width=True)

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
        use_container_width=True,
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
