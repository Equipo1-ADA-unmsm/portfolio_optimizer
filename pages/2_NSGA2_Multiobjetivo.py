"""
Módulo 2 — NSGA-II Multiobjetivo
================================
Optimización bi-objetivo de portafolio (maximizar retorno, minimizar riesgo)
con el algoritmo genético NSGA-II (DEAP). Genera el frente de Pareto frente a
la frontera de Markowitz, 3 portafolios representativos, la evolución del
hypervolume y la simulación de riqueza del portafolio GA.

Parámetros base (tickers, fechas, capital) desde st.session_state.
MU (población) y NGEN (generaciones) configurables con sliders.
"""

import io
import random
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from deap import base, creator, tools
from scipy.optimize import minimize
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuración y paleta
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="NSGA-II Multiobjetivo", page_icon="🧬", layout="wide")

# Estilos (paleta, tipografía, ajuste de modo oscuro y renombrado del sidebar)
#   Definidos en estilos.py para reutilizarse igual en todos los módulos.
aplicar_estilos()

DIAS_ANIO, RF, SEMILLA = 252, 0.02, 42

st.markdown(
    "<h1>🧬 Módulo 2 · NSGA-II Multiobjetivo</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# NUEVO: SIDEBAR — Configuración de Parámetros (Idéntico a app.py)
# --------------------------------------------------------------------------- #
TICKERS_DEFAULT = "FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"
FECHA_INI_DEFAULT = dt.date(2015, 1, 1)
FECHA_FIN_DEFAULT = dt.date(2024, 12, 31)
CAPITAL_DEFAULT = 100_000

with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0'>⚙️ Parámetros</h2>",
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

# ---------------------------------------------------------------------------
# Sincronización y reactividad con session_state 
# ---------------------------------------------------------------------------

tickers_lista = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.session_state["tickers_raw"] = tickers_input
st.session_state["tickers"] = tickers_lista
st.session_state["fecha_ini"] = fecha_ini
st.session_state["fecha_fin"] = fecha_fin
st.session_state["capital"] = int(capital)
st.session_state["max_cash"] = float(MAX_CASH)
tickers_validos = st.session_state.get("tickers", ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"])

if ejecutar:
    st.session_state["analisis_ejecutado"] = True

# Validaciones en el sidebar
if fecha_ini >= fecha_fin:
    st.sidebar.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
if not tickers_lista:
    st.sidebar.error("⚠️ Ingresa al menos un ticker.")
if capital <= 0:
    st.sidebar.error("⚠️ El capital debe ser mayor que 0.")

# --------------------------------------------------------------------------- #
# Sliders del algoritmo
# --------------------------------------------------------------------------- #
col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
with col_s1:
    MU_POP = st.slider("Tamaño de población (MU)", 50, 300, 100, step=10)
with col_s2:
    NGEN = st.slider("Número de generaciones (NGEN)", 30, 200, 80, step=10)
with col_s3:
    st.write("")
    st.write("")
    ejecutar = st.button("🧬 Evolucionar")

# --------------------------------------------------------------------------- #
# Descarga de datos (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cargar_datos(tickers, inicio, fin):
    datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
    precios = datos["Close"]
    if isinstance(precios, pd.Series):
        precios = precios.to_frame()
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)
    precios = precios.dropna(how="all").dropna()
    return precios

# --------------------------------------------------------------------------- #
# Calculador de Hypervolume 2D (Minimización de f0 y f1)
# --------------------------------------------------------------------------- #
def calcular_hv_2d(fitnesses, ref_point):
    """
    Calcula el Hypervolume 2D para un conjunto de puntos de fitness a minimizar.
    fitnesses: lista de tuplas/listas (f0, f1)
    ref_point: tupla/lista (r0, r1) que actúa como punto de referencia superior.
    """
    valid_pts = [p for p in fitnesses if p[0] < ref_point[0] and p[1] < ref_point[1]]
    if not valid_pts:
        return 0.0
    # Ordenar por f0 ascendente. Si hay empates, por f1 descendente
    valid_pts = sorted(valid_pts, key=lambda x: (x[0], x[1]))
    
    # Filtrar puntos dominados
    filtered = []
    for p in valid_pts:
        if not filtered or p[1] < filtered[-1][1]:
            filtered.append(p)
            
    if not filtered:
        return 0.0
        
    r0, r1 = ref_point
    x0, y0 = filtered[0]
    hv = (r0 - x0) * (r1 - y0)
    for i in range(1, len(filtered)):
        xi, yi = filtered[i]
        y_prev = filtered[i-1][1]
        hv += (r0 - xi) * (y_prev - yi)
    return hv

# --------------------------------------------------------------------------- #
# Configuración DEAP
# --------------------------------------------------------------------------- #
def construir_toolbox(mu_vec, Sigma, N, max_cash):
    if hasattr(creator, "FitnessMO"):
        del creator.FitnessMO
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMO", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMO)

    def decodificar(ind):
        w = np.clip(np.array(ind, dtype=float), 0, None)
        s = w.sum()
        if s > 1e-10:
            w = w / s
        else:
            w = np.ones(N) / N
            
        # 2. RESTRICCIÓN DE EFECTIVO
        # El activo CASH es el último (índice -1). Si supera el límite, lo reparamos.
        if w[-1] > max_cash:
            exceso = w[-1] - max_cash
            w[-1] = max_cash
            
            # Redistribuir el exceso proporcionalmente entre las empresas mineras
            suma_acciones = w[:-1].sum()
            if suma_acciones > 1e-10:
                w[:-1] += exceso * (w[:-1] / suma_acciones)
            else:
                # Caso extremo donde todas las acciones eran cero
                w[:-1] += exceso / (N - 1)
                
        return w

    def evaluar(ind):
        w = decodificar(ind)
        ret = w @ mu_vec
        vol = np.sqrt(w @ Sigma @ w)
        return (-ret, vol)

    tb = base.Toolbox()
    tb.register("attr_float", random.random)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_float, n=N)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", evaluar)
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
    tb.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1.0 / N)
    tb.register("select", tools.selNSGA2)
    return tb, decodificar

# --------------------------------------------------------------------------- #
# Ejecución del algoritmo evolutivo
# --------------------------------------------------------------------------- #
if ejecutar:
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    precios = cargar_datos(tickers_lista, fecha_ini, fecha_fin)
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

    tb, decodificar = construir_toolbox(mu_vec, Sigma, N, MAX_CASH)

    # Población inicial
    pop = tb.population(n=MU_POP)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    pop = tb.select(pop, MU_POP)

    # Definir punto de referencia de Hypervolume basado en población inicial
    r0_ref = max(ind.fitness.values[0] for ind in pop) + 0.05
    r1_ref = max(ind.fitness.values[1] for ind in pop) + 0.05
    ref_point = (max(r0_ref, 1.0), max(r1_ref, 1.0))

    CXPB, MUTPB = 0.9, 0.2
    barra = st.progress(0, text="Evolucionando población NSGA-II...")
    hypervolumes = []

    for gen in range(NGEN):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [tb.clone(ind) for ind in offspring]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB:
                tb.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
        for ind in offspring:
            if random.random() < MUTPB:
                tb.mutate(ind)
                del ind.fitness.values

        for ind in [x for x in offspring if not x.fitness.valid]:
            ind.fitness.values = tb.evaluate(ind)

        pop = tb.select(pop + offspring, MU_POP)
        
        # Calcular Hypervolume de la generación
        frente_gen = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        fits_gen = [ind.fitness.values for ind in frente_gen]
        hv_val = calcular_hv_2d(fits_gen, ref_point)
        hypervolumes.append(hv_val)
        
        barra.progress((gen + 1) / NGEN, text=f"Generación {gen + 1}/{NGEN}")

    barra.empty()

    # Frente Pareto final
    frente_final = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pts = np.array([ind.fitness.values for ind in frente_final])
    pts[:, 0] *= -1  # -retorno -> retorno
    orden = np.argsort(pts[:, 1])
    frente_final = [frente_final[i] for i in orden]
    pts = pts[orden]

    # Pesos decodificados de todo el frente
    pesos_frente = [decodificar(ind) for ind in frente_final]

    sharpe_frente = pts[:, 0] / pts[:, 1]
    idx_best = int(np.argmax(sharpe_frente))
    i_cons = int(np.argmin(pts[:, 1]))   # mínimo riesgo
    i_agr = int(np.argmax(pts[:, 0]))    # máximo retorno

    # ---- Frontera de Markowitz para comparación ----
    def min_var_ret(objetivo):
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, o=objetivo: w @ mu_vec - o},
        ]

        limites_mk = [(0.0, 1.0)] * (N - 1) + [(0.0, MAX_CASH)]
        
        r = minimize(lambda w: np.sqrt(w @ Sigma @ w), np.ones(N) / N,
                     method="SLSQP", bounds=limites_mk, constraints=cons)
        return np.sqrt(r.x @ Sigma @ r.x) if r.success else np.nan

    rets_mk = np.linspace(mu_vec.min(), mu_vec.max(), 200)
    vols_mk = np.array([min_var_ret(r) for r in rets_mk])

    # Simulación de riqueza GA (máx Sharpe)
    ret_simples = precios.pct_change().dropna()
    ret_simples['CASH'] = RF / DIAS_ANIO
    w_ga = pesos_frente[idx_best]

    riqueza_bh = [capital]
    w_t = w_ga.copy()
    for i in range(len(ret_simples)):
        r = ret_simples.iloc[i].values
        riqueza_bh.append(riqueza_bh[-1] * (1 + w_t @ r))
        w_t = w_t * (1 + r)
        w_t /= w_t.sum()

    riqueza_reb = [capital]
    w_t = w_ga.copy()
    ult_mes = ret_simples.index[0].month
    for i in range(len(ret_simples)):
        r = ret_simples.iloc[i].values
        riqueza_reb.append(riqueza_reb[-1] * (1 + w_t @ r))
        if ret_simples.index[i].month != ult_mes:
            w_t = w_ga.copy()
            ult_mes = ret_simples.index[i].month
        else:
            w_t = w_t * (1 + r)
            w_t /= w_t.sum()

    fechas_str = [str(f.date()) for f in ([precios.index[0]] + list(ret_simples.index))]

    # Guardar en st.session_state
    st.session_state["nsga2_pts"] = pts
    st.session_state["nsga2_pesos_frente"] = pesos_frente
    st.session_state["nsga2_sharpe_frente"] = sharpe_frente
    st.session_state["nsga2_idx_best"] = idx_best
    st.session_state["nsga2_i_cons"] = i_cons
    st.session_state["nsga2_i_agr"] = i_agr
    st.session_state["nsga2_tickers_optimizacion"] = tickers_optimizacion
    st.session_state["nsga2_vols_mk"] = vols_mk
    st.session_state["nsga2_rets_mk"] = rets_mk
    st.session_state["nsga2_hypervolumes"] = hypervolumes
    st.session_state["nsga2_riqueza_bh"] = riqueza_bh
    st.session_state["nsga2_riqueza_reb"] = riqueza_reb
    st.session_state["nsga2_fechas_str"] = fechas_str
    st.session_state["nsga2_ejecutado"] = True

    # Guardar para el módulo de Comparación
    st.session_state["nsga2_pesos"] = dict(zip(tickers_optimizacion, w_ga.tolist()))
    st.session_state["nsga2_metricas"] = {
        "retorno": float(pts[idx_best, 0]),
        "volatilidad": float(pts[idx_best, 1]),
        "sharpe": float(sharpe_frente[idx_best]),
        "riqueza_bh": float(riqueza_bh[-1]),
        "riqueza_reb": float(riqueza_reb[-1]),
    }

# --------------------------------------------------------------------------- #
# Renderizar UI con datos de session_state si está ejecutado
# --------------------------------------------------------------------------- #
if st.session_state.get("nsga2_ejecutado"):
    pts = st.session_state["nsga2_pts"]
    pesos_frente = st.session_state["nsga2_pesos_frente"]
    sharpe_frente = st.session_state["nsga2_sharpe_frente"]
    idx_best = st.session_state["nsga2_idx_best"]
    i_cons = st.session_state["nsga2_i_cons"]
    i_agr = st.session_state["nsga2_i_agr"]
    tickers_optimizacion = st.session_state["nsga2_tickers_optimizacion"]
    vols_mk = st.session_state["nsga2_vols_mk"]
    rets_mk = st.session_state["nsga2_rets_mk"]
    hypervolumes = st.session_state["nsga2_hypervolumes"]
    riqueza_bh = st.session_state["nsga2_riqueza_bh"]
    riqueza_reb = st.session_state["nsga2_riqueza_reb"]
    fechas_str = st.session_state["nsga2_fechas_str"]

    st.success(f"✅ Frente de Pareto: {len(pesos_frente)} portafolios no dominados.")

    # Tarjetas métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno (Máx Sharpe GA)", f"{pts[idx_best, 0]:.2%}")
    c2.metric("Volatilidad", f"{pts[idx_best, 1]:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe_frente[idx_best]:.3f}")

    st.markdown("---")

    # Gráfico Pareto interactivo
    st.markdown("#### Frente de Pareto NSGA-II vs. Frontera de Markowitz")

    hover_text = []
    for w in pesos_frente:
        detalle = "<br>".join(
            f"{t}: {wi*100:.1f}%" for t, wi in zip(tickers_optimizacion, w) if wi > 0.01
        )
        hover_text.append(detalle)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pts[:, 1] * 100, y=pts[:, 0] * 100, mode="markers",
        marker=dict(size=9, color=sharpe_frente, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe")),
        text=hover_text,
        hovertemplate="σ: %{x:.2f}%<br>E(R): %{y:.2f}%<br>%{text}<extra></extra>",
        name="Frente Pareto (NSGA-II)",
    ))
    mask = ~np.isnan(vols_mk)
    fig.add_trace(go.Scatter(
        x=vols_mk[mask] * 100, y=rets_mk[mask] * 100, mode="lines",
        line=dict(color=GRANATE, dash="dash", width=2), name="Frontera Markowitz",
    ))
    fig.add_trace(go.Scatter(
        x=[pts[idx_best, 1] * 100], y=[pts[idx_best, 0] * 100], mode="markers",
        marker=dict(size=20, color=GRANATE, symbol="star",
                    line=dict(color="black", width=1)),
        name=f"Máx Sharpe ({sharpe_frente[idx_best]:.3f})",
    ))
    fig.update_layout(
        xaxis_title="Riesgo — Volatilidad anual σ (%)",
        yaxis_title="Retorno esperado anual E(R) (%)",
        legend=dict(x=0.01, y=0.99), height=520,
        margin=dict(t=20, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3 portafolios representativos — pie charts
    st.markdown("#### Portafolios representativos del frente")
    perfiles = {"Conservador": i_cons, "Máx Sharpe": idx_best, "Agresivo": i_agr}
    paleta = [AZUL, GRANATE, DORADO, "#4472C4", "#A6A6A6", "#2E7D32"]
    cols = st.columns(3)

    for col, (nombre, idx) in zip(cols, perfiles.items()):
        with col:
            w = pesos_frente[idx]
            df_w = pd.DataFrame({"Activo": tickers_optimizacion, "Peso": w})
            df_w = df_w[df_w["Peso"] > 0.01]
            fig_p = px.pie(df_w, names="Activo", values="Peso", hole=0.35,
                           color_discrete_sequence=paleta)
            fig_p.update_traces(textposition="inside", textinfo="percent+label")
            fig_p.update_layout(
                showlegend=False, height=280, margin=dict(t=40, b=10, l=10, r=10),
                title=dict(
                    text=f"<b>{nombre}</b><br>Ret {pts[idx,0]*100:.1f}% · "
                         f"Vol {pts[idx,1]*100:.1f}%",
                    font=dict(size=13), x=0.5),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")

    # Evolución del hypervolume (Convergencia)
    st.markdown("#### Evolución del Hypervolume por Generación (Convergencia)")
    fig_hv = px.line(
        x=list(range(1, len(hypervolumes) + 1)), y=hypervolumes,
        labels={"x": "Generación", "y": "Hypervolume"},
        title="Convergencia del algoritmo genético multiobjetivo"
    )
    fig_hv.update_traces(line=dict(color=DORADO, width=2.5))
    fig_hv.update_layout(height=350, margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig_hv, use_container_width=True)

    st.markdown("---")

    # Simulación de riqueza
    st.markdown("#### Evolución de la riqueza ($) — portafolio GA (máx Sharpe)")
    df_wealth = pd.DataFrame(
        {"GA Buy & Hold": riqueza_bh, "GA Rebalanceado mensual": riqueza_reb},
        index=pd.to_datetime(fechas_str),
    )
    st.line_chart(df_wealth)

    cf1, cf2 = st.columns(2)
    cf1.metric("Valor final · GA Buy & Hold", f"${riqueza_bh[-1]:,.0f}")
    cf2.metric("Valor final · GA Rebalanceado", f"${riqueza_reb[-1]:,.0f}")

    st.markdown("---")

    # Descarga Excel
    st.markdown("#### Frente de Pareto completo")
    filas = []
    for k, w in enumerate(pesos_frente):
        fila = {"Portafolio": k + 1,
                "Retorno_%": pts[k, 0] * 100,
                "Volatilidad_%": pts[k, 1] * 100,
                "Sharpe": sharpe_frente[k]}
        fila.update({t: wi for t, wi in zip(tickers_optimizacion, w)})
        filas.append(fila)
    df_pareto = pd.DataFrame(filas)

    st.dataframe(df_pareto, use_container_width=True, height=300)

    # 3 portafolios representativos para Excel
    df_representativos = pd.DataFrame([
        {"Perfil": "Conservador", "Retorno_%": pts[i_cons, 0]*100, "Volatilidad_%": pts[i_cons, 1]*100, "Sharpe": sharpe_frente[i_cons], **dict(zip(tickers_optimizacion, pesos_frente[i_cons]))},
        {"Perfil": "Máximo Sharpe", "Retorno_%": pts[idx_best, 0]*100, "Volatilidad_%": pts[idx_best, 1]*100, "Sharpe": sharpe_frente[idx_best], **dict(zip(tickers_optimizacion, pesos_frente[idx_best]))},
        {"Perfil": "Agresivo", "Retorno_%": pts[i_agr, 0]*100, "Volatilidad_%": pts[i_agr, 1]*100, "Sharpe": sharpe_frente[i_agr], **dict(zip(tickers_optimizacion, pesos_frente[i_agr]))},
    ])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_pareto.to_excel(writer, index=False, sheet_name="Frente_Pareto")
        df_representativos.to_excel(writer, index=False, sheet_name="Portafolios_Representativos")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Descargar frente Pareto y representativos (Excel)",
        data=buffer,
        file_name="frente_pareto_nsga2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("👆 Ajusta **MU** y **NGEN** y pulsa **Evolucionar** para correr el NSGA-II.")

st.markdown(
    "<div class='disclaimer' style='margin-top:1rem'>⚠️ <b>Aviso:</b> Los datos son "
    "simulaciones con fines académicos y no constituyen asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
