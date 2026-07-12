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

import bootstrap  # noqa: F401 — debe ir antes de cualquier import de numpy/scipy (ver bootstrap.py)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from deap import base, creator, tools
from scipy.optimize import minimize
import streamlit as st

from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
from sidebar import renderizar_sidebar
from datos import descargar_precios, TTL_PRECIOS_SEGUNDOS

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
tickers_validos = st.session_state.get("tickers", ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"])

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
    # "🔄 Forzar recálculo" (en el sidebar) también debe disparar una nueva
    # evolución aquí, no solo invalidar la caché — de lo contrario el
    # usuario tendría que pulsar dos botones para lograrlo.
    ejecutar = st.button("🧬 Evolucionar") or forzar_recalculo

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
# Parte cacheada y barata de NSGA-II — descarga, retornos y frontera de
# Markowitz de referencia (NO incluye la evolución genética)
#   Antes, esto vivía en la misma función cacheada que corría las MU_POP x
#   NGEN generaciones, cacheada por los 7 parámetros juntos (incluyendo
#   MU_POP y NGEN). Eso significaba que mover el slider de generaciones, por
#   ejemplo, invalidaba también la frontera de Markowitz de comparación (200
#   optimizaciones más) aunque esa frontera no dependa en nada de la
#   población ni de las generaciones. Separarla aquí evita ese recálculo
#   innecesario Y permite que la evolución (la parte que sí vale la pena
#   animar) se corra aparte, generación a generación.
#   TTL: mismo TTL_PRECIOS_SEGUNDOS que descargar_precios() en datos.py.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=TTL_PRECIOS_SEGUNDOS)
def calcular_nsga2_datos(tickers, inicio, fin, capital, max_cash):
    precios, tickers_descartados = descargar_precios(tickers, inicio, fin)

    tickers_validos = list(precios.columns)
    retornos = np.log(precios / precios.shift(1)).dropna()
    retornos['CASH'] = RF / DIAS_ANIO
    tickers_optimizacion = list(retornos.columns)
    N = len(tickers_optimizacion)
    mu_vec = retornos.mean().values * DIAS_ANIO
    Sigma = retornos.cov().values * DIAS_ANIO

    # ---- Frontera de Markowitz para comparación (no depende del GA) ----
    def min_var_ret(objetivo):
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, o=objetivo: w @ mu_vec - o},
        ]
        limites_mk = [(0.0, 1.0)] * (N - 1) + [(0.0, max_cash)]
        r = minimize(lambda w: np.sqrt(w @ Sigma @ w), np.ones(N) / N,
                     method="SLSQP", bounds=limites_mk, constraints=cons)
        return np.sqrt(r.x @ Sigma @ r.x) if r.success else np.nan

    rets_mk = np.linspace(mu_vec.min(), mu_vec.max(), 200)
    vols_mk = np.array([min_var_ret(r) for r in rets_mk])

    # Retornos simples y fechas, para la simulación de riqueza posterior
    ret_simples = precios.pct_change().dropna()
    ret_simples['CASH'] = RF / DIAS_ANIO
    fechas_str = [str(f.date()) for f in ([precios.index[0]] + list(ret_simples.index))]

    return {
        "tickers_descartados": tickers_descartados,
        "tickers_optimizacion": tickers_optimizacion,
        "N": N,
        "mu_vec": mu_vec,
        "Sigma": Sigma,
        "vols_mk": vols_mk,
        "rets_mk": rets_mk,
        "ret_simples": ret_simples,
        "fechas_str": fechas_str,
    }


# --------------------------------------------------------------------------- #
# Evolución NSGA-II — generador NO cacheado con st.cache_data a propósito
#   (un generador no se puede cachear/reanudar con st.cache_data, solo
#   valores de retorno completos). Cede el control (yield) después de CADA
#   generación con el frente de Pareto y el hypervolume acumulado hasta ese
#   punto, para que la página pueda redibujar ambos gráficos en vivo
#   mientras el algoritmo genético evoluciona de verdad — no después, con
#   el resultado ya terminado. La caché de este resultado final se maneja
#   a mano en session_state (ver "_nsga2_cache" más abajo), igual que se
#   hizo con la frontera eficiente en el módulo de Markowitz.
# --------------------------------------------------------------------------- #
def generar_nsga2(mu_vec, Sigma, N, max_cash, mu_pop, ngen, semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)

    tb, decodificar = construir_toolbox(mu_vec, Sigma, N, max_cash)

    pop = tb.population(n=mu_pop)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    pop = tb.select(pop, mu_pop)

    r0_ref = max(ind.fitness.values[0] for ind in pop) + 0.05
    r1_ref = max(ind.fitness.values[1] for ind in pop) + 0.05
    ref_point = (max(r0_ref, 1.0), max(r1_ref, 1.0))

    CXPB, MUTPB = 0.9, 0.2
    hypervolumes = []

    for gen in range(ngen):
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

        pop = tb.select(pop + offspring, mu_pop)

        # Frente de Pareto y hypervolume de ESTA generación
        frente_gen = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        fits_gen = [ind.fitness.values for ind in frente_gen]
        hv_val = calcular_hv_2d(fits_gen, ref_point)
        hypervolumes.append(hv_val)

        pts_gen = np.array(fits_gen)
        pts_gen[:, 0] *= -1  # -retorno -> retorno

        # Se cede `pop` y `decodificar` en cada iteración para que, al
        # terminar el for (última generación), el llamador se quede con la
        # población final y pueda decodificarla sin tener que volver a
        # llamar a construir_toolbox() ni recrear las clases de DEAP.
        yield gen, ngen, pts_gen, list(hypervolumes), pop, decodificar


def construir_fig_pareto(pts, vols_mk, rets_mk, sharpe_frente=None, hover_text=None, idx_best=None):
    """Arma la figura de Plotly del frente de Pareto vs. la frontera de
    Markowitz. Se usa TANTO en cada frame de la animación en vivo (sin
    `sharpe_frente`/`hover_text`/`idx_best`, para no pagar el costo de
    decodificar cada individuo del frente en cada generación) COMO en el
    resultado final ya completo (con esos 3 datos, igual que antes)."""
    fig = go.Figure()
    if sharpe_frente is not None:
        marker = dict(size=9, color=sharpe_frente, colorscale="Viridis",
                      showscale=True, colorbar=dict(title="Sharpe"))
    else:
        marker = dict(size=9, color=AZUL)

    fig.add_trace(go.Scatter(
        x=pts[:, 1] * 100, y=pts[:, 0] * 100, mode="markers",
        marker=marker,
        text=hover_text,
        hovertemplate=("σ: %{x:.2f}%<br>E(R): %{y:.2f}%<br>%{text}<extra></extra>"
                       if hover_text is not None else
                       "σ: %{x:.2f}%<br>E(R): %{y:.2f}%<extra></extra>"),
        name="Frente Pareto (NSGA-II)",
    ))
    mask = ~np.isnan(vols_mk)
    fig.add_trace(go.Scatter(
        x=vols_mk[mask] * 100, y=rets_mk[mask] * 100, mode="lines",
        line=dict(color=GRANATE, dash="dash", width=2), name="Frontera Markowitz",
    ))
    if idx_best is not None:
        fig.add_trace(go.Scatter(
            x=[pts[idx_best, 1] * 100], y=[pts[idx_best, 0] * 100], mode="markers",
            marker=dict(size=20, color=GRANATE, symbol="star",
                        line=dict(color="black", width=1)),
            name="Máx Sharpe",
        ))
    fig.update_layout(
        xaxis_title="Riesgo — Volatilidad anual σ (%)",
        yaxis_title="Retorno esperado anual E(R) (%)",
        legend=dict(x=0.01, y=0.99), height=520,
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


def construir_fig_hv(hypervolumes):
    """Arma la figura de Plotly del hypervolume acumulado (usada tanto en
    cada frame de la animación como en el resultado final)."""
    fig_hv = px.line(
        x=list(range(1, len(hypervolumes) + 1)), y=hypervolumes,
        labels={"x": "Generación", "y": "Hypervolume"},
    )
    fig_hv.update_traces(line=dict(color=DORADO, width=2.5))
    fig_hv.update_layout(height=350, margin=dict(t=20, b=40, l=40, r=20))
    return fig_hv


# --------------------------------------------------------------------------- #
# Ejecución del algoritmo evolutivo — se dispara al pulsar "🧬 Evolucionar"
# (o "🔄 Forzar recálculo", que además invalida la caché antes de evolucionar)
# --------------------------------------------------------------------------- #
if ejecutar:
    if forzar_recalculo:
        descargar_precios.clear()
        calcular_nsga2_datos.clear()
        # La evolución NSGA-II ya no vive en un @st.cache_data (ver nota
        # junto a generar_nsga2): su caché es manual, en session_state, así
        # que "Forzar recálculo" también debe vaciarla para garantizar que
        # se vuelva a animar desde cero.
        st.session_state["_nsga2_cache"] = {}
        st.caption("🔄 Forzando recálculo completo (caché descartada).")

    # Guardamos SOLO los argumentos con los que se llama a calcular_nsga2_datos()
    # y generar_nsga2(), no su resultado (frente de Pareto completo,
    # hypervolumes por generación, series de riqueza, etc.): ese resultado ya
    # vive cacheado (uno en st.cache_data, el otro a mano en session_state)
    # bajo esta misma combinación de parámetros. Guardar solo esta tupla de 7
    # valores (en vez de ~14 claves con datos pesados) reduce bastante el
    # footprint de memoria por sesión.
    st.session_state["nsga2_calc_args"] = (
        tuple(tickers_lista), str(fecha_ini), str(fecha_fin), float(capital), float(MAX_CASH),
        int(MU_POP), int(NGEN),
    )
    st.session_state["nsga2_ejecutado"] = True

# --------------------------------------------------------------------------- #
# Renderizar UI
#   1) calcular_nsga2_datos() — parte barata y cacheada (descarga, mu/Sigma,
#      frontera de Markowitz de referencia). Cache-hit instantáneo si los
#      parámetros de mercado no cambiaron.
#   2) La evolución GA en sí: si ya se corrió antes con ESTOS MISMOS 7
#      parámetros (incluyendo MU_POP/NGEN), se reutiliza de la caché manual
#      "_nsga2_cache" sin animar de nuevo. Si es una corrida nueva, se anima
#      en vivo: el frente de Pareto y el hypervolume se redibujan generación
#      a generación, mientras el algoritmo genético evoluciona de verdad.
# --------------------------------------------------------------------------- #
if st.session_state.get("nsga2_ejecutado"):
    tickers_calc, fecha_ini_calc, fecha_fin_calc, capital_calc, max_cash_calc, mu_pop_calc, ngen_calc = (
        st.session_state["nsga2_calc_args"]
    )

    with st.spinner("Descargando datos y preparando el problema..."):
        datos_base = calcular_nsga2_datos(
            tickers_calc, fecha_ini_calc, fecha_fin_calc, capital_calc, max_cash_calc,
        )

    tickers_descartados = datos_base["tickers_descartados"]
    tickers_optimizacion = datos_base["tickers_optimizacion"]
    N = datos_base["N"]
    mu_vec = datos_base["mu_vec"]
    Sigma = datos_base["Sigma"]
    vols_mk = datos_base["vols_mk"]
    rets_mk = datos_base["rets_mk"]
    ret_simples = datos_base["ret_simples"]
    fechas_str = datos_base["fechas_str"]

    ga_cache = st.session_state.setdefault("_nsga2_cache", {})

    st.markdown("#### Frente de Pareto NSGA-II vs. Frontera de Markowitz")
    placeholder_pareto_titulo = st.empty()
    placeholder_pareto = st.empty()
    st.markdown("#### Evolución del Hypervolume por Generación (Convergencia)")
    placeholder_hv = st.empty()

    if st.session_state["nsga2_calc_args"] in ga_cache:
        # Cache-hit: ya se evolucionó antes con estos mismos 7 parámetros
        # (p. ej. el usuario navegó a otra página y volvió). No se anima de
        # nuevo, se muestra el resultado final directamente.
        resultado_ga = ga_cache[st.session_state["nsga2_calc_args"]]
        placeholder_pareto_titulo.empty()
    else:
        # Cache-miss: corrida genuinamente nueva. Se anima generación a
        # generación mientras el algoritmo evoluciona.
        paso = max(1, ngen_calc // 40)  # ~40 actualizaciones en total
        pop_final, decodificar_final, hv_final = None, None, []
        for gen, total_gen, pts_gen, hv_historial, pop, decodificar in generar_nsga2(
            mu_vec, Sigma, N, max_cash_calc, mu_pop_calc, ngen_calc,
        ):
            pop_final, decodificar_final, hv_final = pop, decodificar, hv_historial
            if gen % paso == 0 or gen == total_gen - 1:
                placeholder_pareto_titulo.caption(f"🧬 Evolucionando… generación {gen + 1}/{total_gen}")
                placeholder_pareto.plotly_chart(
                    construir_fig_pareto(pts_gen, vols_mk, rets_mk),
                    width='stretch', key=f"pareto_frame_{gen}",
                )
                placeholder_hv.plotly_chart(
                    construir_fig_hv(hv_historial),
                    width='stretch', key=f"hv_frame_{gen}",
                )
        placeholder_pareto_titulo.empty()

        # La evolución terminó: se arma el resultado final a partir de la
        # última población (pop_final) y se guarda en la caché manual.
        frente_final = tools.sortNondominated(pop_final, len(pop_final), first_front_only=True)[0]
        pts = np.array([ind.fitness.values for ind in frente_final])
        pts[:, 0] *= -1  # -retorno -> retorno
        orden = np.argsort(pts[:, 1])
        frente_final = [frente_final[i] for i in orden]
        pts = pts[orden]

        pesos_frente = [decodificar_final(ind) for ind in frente_final]
        sharpe_frente = pts[:, 0] / pts[:, 1]
        idx_best = int(np.argmax(sharpe_frente))
        i_cons = int(np.argmin(pts[:, 1]))
        i_agr = int(np.argmax(pts[:, 0]))
        w_ga = pesos_frente[idx_best]

        # Simulación de riqueza GA (máx Sharpe) — rápida, no se anima.
        riqueza_bh = [capital_calc]
        w_t = w_ga.copy()
        for i in range(len(ret_simples)):
            r = ret_simples.iloc[i].values
            riqueza_bh.append(riqueza_bh[-1] * (1 + w_t @ r))
            w_t = w_t * (1 + r)
            w_t /= w_t.sum()

        riqueza_reb = [capital_calc]
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

        resultado_ga = {
            "pts": pts,
            "pesos_frente": pesos_frente,
            "sharpe_frente": sharpe_frente,
            "idx_best": idx_best,
            "i_cons": i_cons,
            "i_agr": i_agr,
            "hypervolumes": hv_final,
            "riqueza_bh": riqueza_bh,
            "riqueza_reb": riqueza_reb,
            "w_ga": w_ga,
        }
        ga_cache[st.session_state["nsga2_calc_args"]] = resultado_ga

    pts = resultado_ga["pts"]
    pesos_frente = resultado_ga["pesos_frente"]
    sharpe_frente = resultado_ga["sharpe_frente"]
    idx_best = resultado_ga["idx_best"]
    i_cons = resultado_ga["i_cons"]
    i_agr = resultado_ga["i_agr"]
    hypervolumes = resultado_ga["hypervolumes"]
    riqueza_bh = resultado_ga["riqueza_bh"]
    riqueza_reb = resultado_ga["riqueza_reb"]
    w_ga = resultado_ga["w_ga"]

    # Guardar para el módulo de Comparación (pequeño: dicts de floats, no
    # arrays ni el frente completo — barato de recalcular en cada rerun).
    # Se usan los parámetros REALMENTE calculados (tickers_calc, etc.), no
    # los valores actuales del sidebar, que podrían haber cambiado desde
    # la última corrida sin que el usuario vuelva a pulsar "Evolucionar".
    st.session_state["nsga2_pesos"] = dict(zip(tickers_optimizacion, w_ga.tolist()))
    st.session_state["nsga2_metricas"] = {
        "retorno": float(pts[idx_best, 0]),
        "volatilidad": float(pts[idx_best, 1]),
        "sharpe": float(sharpe_frente[idx_best]),
        "riqueza_bh": float(riqueza_bh[-1]),
        "riqueza_reb": float(riqueza_reb[-1]),
    }
    st.session_state["nsga2_params"] = (
        tickers_calc, fecha_ini_calc, fecha_fin_calc, capital_calc, max_cash_calc,
    )

    if tickers_descartados:
        st.warning(
            "⚠️ Se descartaron los siguientes tickers por no tener datos válidos en "
            f"Yahoo Finance para el rango de fechas seleccionado: {', '.join(tickers_descartados)}. "
            "El análisis continuó con el resto del universo."
        )

    st.success(f"✅ Frente de Pareto: {len(pesos_frente)} portafolios no dominados.")

    # Tarjetas métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno (Máx Sharpe GA)", f"{pts[idx_best, 0]:.2%}")
    c2.metric("Volatilidad", f"{pts[idx_best, 1]:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe_frente[idx_best]:.3f}")

    # Redibuja el frente de Pareto y el hypervolume en su versión FINAL
    # completa (con colorbar de Sharpe, detalle de pesos por portafolio en
    # el hover, y la estrella del máximo Sharpe) sobre los mismos
    # placeholders usados para animar — si vino de caché, es la primera vez
    # que se dibujan en este rerun.
    hover_text = []
    for w in pesos_frente:
        detalle = "<br>".join(
            f"{t}: {wi*100:.1f}%" for t, wi in zip(tickers_optimizacion, w) if wi > 0.01
        )
        hover_text.append(detalle)

    placeholder_pareto.plotly_chart(
        construir_fig_pareto(pts, vols_mk, rets_mk, sharpe_frente=sharpe_frente,
                             hover_text=hover_text, idx_best=idx_best),
        width='stretch', key="pareto_final",
    )
    placeholder_hv.plotly_chart(
        construir_fig_hv(hypervolumes), width='stretch', key="hv_final",
    )

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
            st.plotly_chart(fig_p, width='stretch')

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

    st.dataframe(df_pareto, width='stretch', height=300)

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
