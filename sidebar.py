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
    forzar_recalculo = parametros["forzar_recalculo"]

  Para páginas que tienen su propio botón de ejecución específico (p. ej.
  NSGA-II con "🧬 Evolucionar", o DP con "🔁 Ejecutar DP") y no necesitan el
  botón genérico "🚀 Ejecutar Análisis" del sidebar, pasar:

      parametros = renderizar_sidebar(mostrar_boton_ejecutar=False)

  Para páginas de análisis (1-4), que si dependen de que los parámetros
  sean válidos antes de continuar, pasar:

      parametros = renderizar_sidebar(detener_si_invalido=True)

Forzar recálculo:
  Con Markowitz, NSGA-II, DP y Comparación cacheados por parámetros
  (st.cache_data), volver a pulsar el botón de análisis de una página con
  los mismos parámetros ya no recalcula nada — sirve el resultado desde
  caché. Eso es justo lo que se buscaba, pero deja sin forma de decir
  "no confíes en nada cacheado, quiero recalcular todo de cero" (p. ej.
  para verificar reproducibilidad, o porque se sospecha que yfinance
  cambió datos históricos con un ajuste retroactivo dentro de la misma
  ventana de TTL). Por eso el sidebar siempre expone un botón
  "🔄 Forzar recálculo", devuelto como `parametros["forzar_recalculo"]`:
  cada página, al ver que viene en True, se encarga de invalidar (con
  `.clear()`) tanto la caché de `descargar_precios()` como la de su
  propia función de cálculo pesado antes de ejecutar — así el próximo
  cálculo es garantizado un cache-miss real, no solo una repetición de
  los mismos parámetros.

Reiniciar todo:
  A diferencia de "Forzar recálculo" (que solo invalida caché para volver
  a calcular con los MISMOS parámetros), "🗑️ Reiniciar todo" borra por
  completo el estado de la sesión: los parámetros vuelven a sus valores
  por defecto y se pierden los resultados ya calculados en las 4 páginas
  (Markowitz, NSGA-II, DP y lo que Comparación tuviera reutilizado de
  ellas). Pensado para cuando el usuario quiere empezar de cero sin
  recargar el navegador (lo cual, en Streamlit Community Cloud, además
  reutilizaría la misma sesión de servidor y no limpiaría nada por sí
  solo).

  Implementación: los widgets de tickers/fechas/capital/límite de efectivo
  tienen ahora un `key` explícito. Esto es necesario porque un widget de
  Streamlit SIN `key` no se puede "resetear" con `st.session_state.clear()`
  — su valor vive en el estado interno del widget, no en session_state, y
  el argumento `value=` solo se usa la primera vez que el widget se crea.
  Con `key`, el valor del widget SÍ vive en `st.session_state[key]`, así
  que al borrar esa entrada (o toda la sesión) el widget vuelve a
  instanciarse con su valor por defecto en el siguiente rerun.

  Por tratarse de una acción irreversible dentro de la sesión (no hay
  "deshacer"), el botón pide una confirmación en dos pasos antes de
  ejecutar `st.session_state.clear()` + `st.rerun()`.
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
    max_cash, ejecutar, valido, forzar_recalculo.
    """
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom:0'>⚙️ Parámetros</h2>",
                    unsafe_allow_html=True)
        st.caption("Configuración global del análisis")

        tickers_input = st.text_input(
            "Tickers (separados por coma)",
            value=st.session_state.get("tickers_raw", TICKERS_DEFAULT),
            help="Símbolos de Yahoo Finance. Ej: FSM, BHP, BVN",
            key="sb_tickers_input",
        )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fecha_ini = st.date_input(
                "Fecha inicio",
                value=st.session_state.get("fecha_ini", FECHA_INI_DEFAULT),
                min_value=dt.date(2000, 1, 1),
                max_value=dt.date.today(),
                key="sb_fecha_ini",
            )
        with col_f2:
            fecha_fin = st.date_input(
                "Fecha fin",
                value=st.session_state.get("fecha_fin", FECHA_FIN_DEFAULT),
                min_value=dt.date(2000, 1, 1),
                max_value=dt.date.today(),
                key="sb_fecha_fin",
            )

        capital = st.number_input(
            "Capital a invertir (USD)",
            min_value=1_000,
            max_value=100_000_000,
            value=st.session_state.get("capital", CAPITAL_DEFAULT),
            step=1_000,
            format="%d",
            key="sb_capital",
        )

        max_cash = st.slider(
            "Límite máx. Efectivo",
            0.0, 1.0, float(st.session_state.get("max_cash", MAX_CASH_DEFAULT)),
            step=0.05,
            format="%.2f",
            help="Porcentaje máximo del portafolio que puede mantenerse en CASH.",
            key="sb_max_cash",
        )

        ejecutar = False
        if mostrar_boton_ejecutar:
            st.markdown("---")
            ejecutar = st.button("🚀 Ejecutar Análisis")

        # Botón de forzar recálculo — siempre visible, sin importar si esta
        # página muestra o no el botón genérico de arriba (p. ej. el módulo
        # de DP, que tiene su propio botón "🔁 Ejecutar DP" en el cuerpo de
        # la página, también se beneficia de poder invalidar su caché desde
        # aquí). Cada página decide qué caché(s) limpiar al ver este flag.
        st.markdown("---")
        forzar_recalculo = st.button(
            "🔄 Forzar recálculo",
            help=(
                "Descarta cualquier precio o resultado ya cacheado para los "
                "parámetros actuales y vuelve a calcular todo desde cero "
                "(incluye una nueva descarga de precios), en vez de servir "
                "lo que ya se había calculado antes con estos mismos "
                "parámetros."
            ),
        )

        st.markdown("---")
        st.caption("💡 Los parámetros se comparten entre todas las páginas.")

        # ------------------------------------------------------------------- #
        # Reiniciar todo — borra parámetros y resultados de las 4 páginas.
        #   Es una acción irreversible dentro de la sesión (no hay "deshacer":
        #   se pierde cualquier resultado de Markowitz/NSGA-II/DP ya
        #   calculado), así que pide confirmación en dos pasos en vez de
        #   actuar directo al primer clic.
        # ------------------------------------------------------------------- #
        st.markdown("---")
        if st.session_state.get("_confirmar_reinicio"):
            st.warning(
                "⚠️ Esto borrará todos los parámetros (vuelven a sus valores "
                "por defecto) y los resultados ya calculados en las 4 "
                "páginas. ¿Confirmas?"
            )
            col_si, col_no = st.columns(2)
            with col_si:
                confirmar = st.button("✅ Sí, reiniciar", width="stretch")
            with col_no:
                cancelar = st.button("✖️ Cancelar", width="stretch")
            if confirmar:
                st.session_state.clear()
                st.rerun()
            if cancelar:
                st.session_state["_confirmar_reinicio"] = False
                st.rerun()
        else:
            pedir_reinicio = st.button(
                "🗑️ Reiniciar todo",
                help=(
                    "Borra todos los parámetros configurados y los resultados "
                    "ya calculados en Markowitz, NSGA-II, DP y Comparación, "
                    "volviendo la app a su estado inicial."
                ),
            )
            if pedir_reinicio:
                st.session_state["_confirmar_reinicio"] = True
                st.rerun()

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

    # Forzar recálculo implica también ejecutar: no tendría sentido limpiar
    # la caché y no volver a correr el análisis en el mismo clic.
    if forzar_recalculo:
        ejecutar = True

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
        "forzar_recalculo": forzar_recalculo,
    }
