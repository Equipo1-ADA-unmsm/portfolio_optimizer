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

import streamlit as st

from estilos import aplicar_estilos
from sidebar import renderizar_sidebar

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
