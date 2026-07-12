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
    """
    datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
    precios = datos["Close"]

    if isinstance(precios, pd.Series):
        precios = precios.to_frame()
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)

    precios = precios.dropna(axis=1, how="all").ffill().dropna()
    return precios
