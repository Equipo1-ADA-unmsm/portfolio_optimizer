"""
Sidebar de parámetros compartido
=================================
Módulo centralizado para el sidebar de configuración global (tickers, fechas,
capital, límite de efectivo) que se repetía casi idéntico en app.py y en los
4 módulos de pages/.

Por qué existe este archivo:
  Antes cada página definía su propio bloque `with st.sidebar:` con los
  mismos widgets, y sus propias validaciones. Si un default o una validación
  se corregía en una página y no en otra, aparecían inconsistencias (p. ej.
  el slider de "Límite máx. Efectivo" que se olvidó de leer session_state en
  un módulo y se reseteaba solo). Ahora toda página solo necesita:

      from sidebar import renderizar_sidebar
      parametros = renderizar_sidebar()

  y automáticamente comparte los mismos widgets, valores por defecto,
  sincronización con session_state y validaciones.

Validación centralizada (detener_si_invalido):
  Antes, cada una de las 4 páginas de análisis repetía (o directamente
  omitía) su propio chequeo de "¿hay tickers configurados?" antes de
  intentar descargar datos o correr una optimización. Al revisar el
  código se encontró que el módulo 2 (NSGA-II) y el módulo 3 (DP) NO
  tenían ningún chequeo antes de ejecutar — un capital <= 0, fechas
  invertidas o una lista de tickers vacía podían llegar hasta
  `descargar_precios()` o a scipy.optimize sin un mensaje claro primero.
  Ahora el sidebar valida los 3 casos (fechas, tickers, capital) en un
  solo lugar y, si se pide con `detener_si_invalido=True`, corta la
  ejecución con `st.stop()` inmediatamente después de mostrar los
  mensajes de error — antes de que la página siga corriendo con un
  estado inválido. El default es `False` para no romper páginas como
  app.py (el home), que solo *muestra* la configuración actual y no
  hace ninguna descarga ni optimización, por lo que puede seguir
  renderizándose con normalidad aunque el estado sea inválido (el
  usuario ve el aviso en el sidebar y corrige desde ahí).

Uso:
    parametros = renderizar_sidebar()
    tickers_lista = parametros["tickers_lista"]
    fecha_ini     = parametros["fecha_ini"]
    fecha_fin     = parametros["fecha_fin"]
    capital       = parametros["capital"]
    max_cash      = parametros["max_cash"]
    ejecutar      = parametros["ejecutar"]
    valido        = parametros["valido"]

  Para páginas que tienen su propio botón de ejecución específico (p. ej.
  NSGA-II con "🧬 Evolucionar", o DP con "🔁 Ejecutar DP") y no necesitan el
  botón genérico "🚀 Ejecutar Análisis" del sidebar, pasar:

      parametros = renderizar_sidebar(mostrar_boton_ejecutar=False)

  Para páginas de análisis (1-4), que si dependen de que los parámetros
  sean válidos antes de continuar, pasar:

      parametros = renderizar_sidebar(detener_si_invalido=True)
"""

import datetime as dt

import streamlit as st

# --------------------------------------------------------------------------- #
# Valores por defecto — únicos para toda la app
# --------------------------------------------------------------------------- #
TICKERS_DEFAULT = "FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"
FECHA_INI_DEFAULT = dt.date(2015, 1, 1)
FECHA_FIN_DEFAULT = dt.date(2024, 12, 31)
CAPITAL_DEFAULT = 100_000
MAX_CASH_DEFAULT = 0.20


def renderizar_sidebar(mostrar_boton_ejecutar: bool = True,
                        detener_si_invalido: bool = False) -> dict:
    """Dibuja el sidebar de parámetros y devuelve los valores sincronizados.

    Parameters
    ----------
    mostrar_boton_ejecutar:
        Si True (default), muestra el botón genérico "🚀 Ejecutar Análisis".
        Pasar False en páginas que ya tienen su propio botón de ejecución
        específico más abajo en el cuerpo de la página.
    detener_si_invalido:
        Si True, y los parámetros actuales NO pasan las validaciones
        (fechas, tickers, capital), corta la ejecución de la página con
        `st.stop()` justo después de mostrar los mensajes de error en el
        sidebar. Pensado para las páginas de análisis (1-4), que no deben
        seguir corriendo con un estado inválido. Default False para no
        afectar páginas puramente informativas (como el home).

    Returns
    -------
    dict con las claves: tickers_lista, fecha_ini, fecha_fin, capital,
    max_cash, ejecutar, valido.
    """
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

        max_cash = st.slider(
            "Límite máx. Efectivo",
            0.0, 1.0, float(st.session_state.get("max_cash", MAX_CASH_DEFAULT)),
            step=0.05,
            format="%.2f",
            help="Porcentaje máximo del portafolio que puede mantenerse en CASH.",
        )

        ejecutar = False
        if mostrar_boton_ejecutar:
            st.markdown("---")
            ejecutar = st.button("🚀 Ejecutar Análisis")

        st.markdown("---")
        st.caption("💡 Los parámetros se comparten entre todas las páginas.")

    # ----------------------------------------------------------------------- #
    # Sincronización y reactividad con session_state
    # ----------------------------------------------------------------------- #
    tickers_lista = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.session_state["tickers_raw"] = tickers_input
    st.session_state["tickers"] = tickers_lista
    st.session_state["fecha_ini"] = fecha_ini
    st.session_state["fecha_fin"] = fecha_fin
    st.session_state["capital"] = int(capital)
    st.session_state["max_cash"] = float(max_cash)

    if ejecutar:
        st.session_state["analisis_ejecutado"] = True

    # ----------------------------------------------------------------------- #
    # Validaciones (se muestran en el sidebar)
    # ----------------------------------------------------------------------- #
    fechas_invalidas = fecha_ini >= fecha_fin
    sin_tickers = not tickers_lista
    capital_invalido = capital <= 0

    if fechas_invalidas:
        st.sidebar.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
    if sin_tickers:
        st.sidebar.error("⚠️ Ingresa al menos un ticker.")
    if capital_invalido:
        st.sidebar.error("⚠️ El capital debe ser mayor que 0.")

    valido = not (fechas_invalidas or sin_tickers or capital_invalido)

    # Si la página lo pidió explícitamente, no seguimos ejecutando su cuerpo
    # con un estado inválido: cortamos aquí, justo después de mostrar los
    # errores arriba, en vez de dejar que la página siga y falle más abajo
    # (p. ej. al llamar a descargar_precios() o a scipy.optimize) con un
    # traceback menos claro para el usuario.
    if detener_si_invalido and not valido:
        st.info("👈 Corrige los parámetros en la barra lateral para continuar.")
        st.stop()

    return {
        "tickers_lista": tickers_lista,
        "fecha_ini": fecha_ini,
        "fecha_fin": fecha_fin,
        "capital": int(capital),
        "max_cash": float(max_cash),
        "ejecutar": ejecutar,
        "valido": valido,
    }
