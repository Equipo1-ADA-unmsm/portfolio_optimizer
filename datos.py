"""
Descarga de datos de mercado (Yahoo Finance)
=============================================
Función centralizada y cacheada para descargar precios de cierre ajustados,
compartida por los 4 módulos de análisis (Markowitz, NSGA-II, DP y
Comparación).

Por qué existe este archivo:
  Cada módulo tenía su propia función de descarga, casi idéntica pero con
  pequeñas diferencias entre sí (algunas no manejaban columnas MultiIndex de
  yfinance, otras no rellenaban huecos con ffill). Al ser funciones Python
  distintas (aunque casi iguales), Streamlit las cacheaba por separado: si
  ya habías descargado precios en el módulo de Markowitz, el módulo NSGA-II
  volvía a pedirlos a Yahoo Finance en vez de reusar esa descarga. Ahora, al
  ser una única función importada en las 4 páginas, la caché de
  `st.cache_data` se comparte entre módulos: el primero en descargar un
  universo de tickers para un rango de fechas beneficia a los demás.

Uso:
    from datos import descargar_precios
    precios = descargar_precios(tickers, fecha_ini, fecha_fin)

Nota sobre el comportamiento unificado:
  Se combinó lo más robusto de las 4 versiones anteriores:
    - Manejo de columnas MultiIndex de yfinance (lo tenían NSGA-II, DP y
      Comparación, pero no Markowitz).
    - Relleno hacia adelante de huecos puntuales con ffill() antes de
      descartar filas (lo tenía Markowitz, pero no los otros 3).
  Antes de esta unificación, Markowitz descargaba con una lógica y los
  otros 3 módulos con otra ligeramente distinta, lo que podía llevar a
  series de precios con un número distinto de filas útiles para el mismo
  universo de tickers y fechas. Con la función única, los 4 módulos parten
  exactamente de los mismos datos — algo especialmente importante para que
  la Comparación sea una comparación justa.

Manejo de errores:
  Si Yahoo Finance no responde (problema de red, servicio caído, etc.) o
  si los tickers/fechas no arrojan ningún dato válido, esta función
  muestra un mensaje de error amigable con `st.error()` y detiene la
  ejecución de la página con `st.stop()`, en vez de dejar que el usuario
  vea un traceback crudo de Python. Antes, cada módulo repetía su propia
  versión de este chequeo (y el módulo de Comparación no tenía ninguno);
  ahora vive en un solo lugar y protege a los 4 módulos por igual.
"""

import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(show_spinner=False)
def descargar_precios(tickers, inicio, fin):
    """Descarga precios de cierre ajustados desde Yahoo Finance.

    Parameters
    ----------
    tickers : list[str] | tuple[str]
        Símbolos a descargar.
    inicio, fin : str | datetime.date
        Rango de fechas.

    Returns
    -------
    pd.DataFrame
        Precios de cierre ajustados. Columnas = tickers con datos válidos
        (se descartan las que vienen enteramente vacías), huecos puntuales
        rellenados hacia adelante, y sin filas NaN remanentes.

        Si la descarga falla o no arroja datos válidos, la función muestra
        un error y detiene la ejecución de la página (no retorna un
        DataFrame vacío a quien la llama).
    """
    try:
        datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
        precios = datos["Close"]

        if isinstance(precios, pd.Series):
            precios = precios.to_frame()
        if isinstance(precios.columns, pd.MultiIndex):
            precios.columns = precios.columns.get_level_values(0)

        precios = precios.dropna(axis=1, how="all").ffill().dropna()
    except Exception:
        st.error(
            "⚠️ No se pudo conectar con Yahoo Finance para descargar los precios. "
            "Esto suele deberse a un problema temporal de red o a que el servicio "
            "no está disponible en este momento. Intenta de nuevo en unos minutos."
        )
        st.stop()

    if precios.empty or precios.shape[1] == 0:
        st.error(
            "⚠️ No se pudieron descargar datos válidos para los tickers indicados. "
            "Verifica que los símbolos existan en Yahoo Finance (ej: FSM, BHP, BVN) "
            "y que el rango de fechas tenga información disponible."
        )
        st.stop()

    return precios
