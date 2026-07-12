"""
Bootstrap de entorno — límite de hilos para BLAS/OpenMP
=========================================================
Este módulo DEBE importarse antes que numpy/scipy en cualquier script de la
app (app.py y cada página de pages/), porque las librerías de álgebra
lineal que usan por debajo (OpenBLAS/MKL) leen estas variables de entorno
una sola vez, en el momento en que numpy se importa por primera vez en
todo el proceso — no en cada rerun de un script de Streamlit.

Por qué es necesario:
  En contenedores con CPU limitada (como Streamlit Community Cloud),
  OpenBLAS/MKL intentan lanzar más hilos de cómputo de los que el
  contenedor realmente tiene disponibles para las operaciones matriciales
  de scipy.optimize (usadas en Markowitz, DP y el NSGA-II interno de
  Comparación). Esto puede corromper memoria y provocar un
  'Segmentation fault' que tumba todo el proceso de Streamlit — un crash
  a nivel de sistema operativo, no una excepción de Python, por lo que NO
  se puede atrapar con try/except ni prevenir desde la lógica de la app.

Importante — por qué vive en su propio archivo y se importa el primero:
  Streamlit puede ejecutar cualquier página como punto de entrada real del
  proceso (por ejemplo, si alguien entra directo a la URL de un módulo en
  vez de pasar por la página de inicio). Por eso NO basta con fijar estas
  variables solo en app.py: hay que garantizar que se fijen ANTES del
  primer `import numpy`, sin importar qué página se ejecute primero. Este
  archivo se importa como la PRIMERA línea en app.py y en las 4 páginas de
  pages/, antes de cualquier import de numpy/scipy/pandas.

Uso (siempre como el primer import del archivo):
    import bootstrap  # noqa: F401  (debe ir antes de "import numpy")

    import numpy as np
    ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
