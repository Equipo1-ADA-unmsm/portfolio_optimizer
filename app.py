"""
Sistema de Optimización de Portafolio
=====================================
Homepage (app.py) — Streamlit

Integra 3 métodos de optimización:
  1. Markowitz (media-varianza)
  2. NSGA-II (algoritmo genético multiobjetivo con DEAP)
  3. Programación Dinámica (backward induction de Bellman)

Datos: Yahoo Finance.  Capital default: USD $100,000.
Tickers default: 5 mineras -> FSM, VOLCABC1.LM, ABX.TO, BVN, BHP.

Requisitos: Python 3.10+, Streamlit 1.x
"""

import datetime as dt
import streamlit as st

from estilos import aplicar_estilos

# --------------------------------------------------------------------------- #
# Configuración general de la página (Pestaña del navegador)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dashboard Principal", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Estilos (paleta, tipografía, tarjetas y ajuste de modo oscuro)
#   Definidos en estilos.py para reutilizarse igual en todos los módulos.
# --------------------------------------------------------------------------- #
aplicar_estilos()

# --------------------------------------------------------------------------- #
# Valores por defecto
# --------------------------------------------------------------------------- #
TICKERS_DEFAULT = "FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"
FECHA_INI_DEFAULT = dt.date(2015, 1, 1)
FECHA_FIN_DEFAULT = dt.date(2024, 12, 31)
CAPITAL_DEFAULT = 100_000

# --------------------------------------------------------------------------- #
# SIDEBAR — Configuración de Parámetros
# --------------------------------------------------------------------------- #
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

    # NUEVO: Slider global de Límite de Efectivo
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

# --------------------------------------------------------------------------- #
# Persistencia en session_state (compartido entre páginas)
# --------------------------------------------------------------------------- #
tickers_lista = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.session_state["tickers_raw"] = tickers_input
st.session_state["tickers"] = tickers_lista
st.session_state["fecha_ini"] = fecha_ini
st.session_state["fecha_fin"] = fecha_fin
st.session_state["capital"] = int(capital)
st.session_state["max_cash"] = float(MAX_CASH)

if ejecutar:
    st.session_state["analisis_ejecutado"] = True

# Validaciones básicas
if fecha_ini >= fecha_fin:
    st.sidebar.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
if not tickers_lista:
    st.sidebar.error("⚠️ Ingresa al menos un ticker.")
if capital <= 0:
    st.sidebar.error("⚠️ El capital debe ser mayor que 0.")

# --------------------------------------------------------------------------- #
# CONTENIDO PRINCIPAL
# --------------------------------------------------------------------------- #
st.markdown(
    "<h1 style='margin-bottom:0'>📈 Sistema de Optimización de Portafolio</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitulo-principal' style='font-size:1.05rem;font-weight:600;margin-top:0.2rem'>"
    "Markowitz · NSGA-II · Programación Dinámica</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Equipo 1 y Docente
# --------------------------------------------------------------------------- #
st.markdown("### 🎓 Equipo 1")

# Listado de integrantes dividido en dos columnas para mejor visibilidad
col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown(
        """
        **Integrantes:**
        * Cordero Alfaro, Renzo Pedro
        * Cansaya Cutipa, Frank Manuel
        * Ccapcha Espinoza, Bruno Rafhael
        * Burga Montesinos, Jeanpiere Jesus
        * Raymondes Peña, Jesús Grabiel
        * Cacya Torocahua, Midwar Jose
        """
    )

with col_eq2:
    st.markdown(
        """
        <br>
        
        * Cencia Pérez, Alvaro Enrique
        * Florencio Valenzuela, David Abraham
        * Jaico Fernandez, Fernando Jose
        * Purisaca Moquillaza, Joseph Francis
        * Galvez Garro, Jorge Luis Junior
        """,
        unsafe_allow_html=True
    )

st.markdown(f"**Docente:** Ernesto Cancho Rodríguez")

st.markdown("---")

st.write(
    "Plataforma de análisis cuantitativo que compara tres enfoques de optimización de "
    "portafolios sobre un universo de activos mineros, usando datos históricos de "
    "**Yahoo Finance**. Configura los parámetros en la barra lateral y navega por los "
    "módulos usando el menú de páginas."
)

# Resumen de la configuración actual
st.markdown("### Configuración actual")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Activos", f"{len(tickers_lista)}")
c2.metric("Capital", f"${int(capital):,.0f}")
c3.metric("Horizonte", f"{fecha_ini.year}–{fecha_fin.year}")
c4.metric("Límite Efectivo", f"{MAX_CASH * 100:.0f}%")

if tickers_lista:
    st.write("**Tickers seleccionados:** " + ", ".join(tickers_lista))

st.markdown("---")

# --------------------------------------------------------------------------- #
# Descripción de los 4 módulos
# --------------------------------------------------------------------------- #
st.markdown("### Módulos del sistema")

modulos = [
    ("1 · Datos & Markowitz",
     "Descarga de precios desde Yahoo Finance, cálculo de retornos y matriz de "
     "covarianzas. Optimización media-varianza y frontera eficiente."),
    ("2 · NSGA-II",
     "Algoritmo genético multiobjetivo (DEAP) que optimiza simultáneamente "
     "rentabilidad y riesgo, generando un frente de Pareto de portafolios."),
    ("3 · Programación Dinámica",
     "Backward induction de Bellman para asignación secuencial de capital, "
     "resolviendo el problema de decisión por etapas."),
    ("4 · Comparación",
     "Contraste de los tres métodos: pesos, retorno esperado, volatilidad y "
     "ratio de Sharpe, con visualizaciones comparativas."),
]

cols = st.columns(4)
for col, (titulo, desc) in zip(cols, modulos):
    with col:
        st.markdown(
            f"<div class='modulo-card'><h4>{titulo}</h4><p>{desc}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.get("analisis_ejecutado"):
    st.success(
        "✅ Parámetros cargados. Abre los módulos en el menú lateral de páginas "
        "para ejecutar cada método."
    )
else:
    st.info("👈 Configura los parámetros y pulsa **Ejecutar Análisis** para comenzar.")

# --------------------------------------------------------------------------- #
# Disclaimer
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown(
    "<div class='disclaimer'>⚠️ <b>Aviso:</b> Los datos son simulaciones con fines "
    "académicos y no constituyen asesoría de inversión.</div>",
    unsafe_allow_html=True,
)
