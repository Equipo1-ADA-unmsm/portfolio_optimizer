"""
Estilos y paleta de colores compartidos
========================================
Módulo centralizado para mantener la MISMA identidad visual (colores,
tipografía, tarjetas, disclaimer, etc.) en app.py y en todas las páginas
del menú lateral (módulos 1-4).

Por qué existe este archivo:
  Antes cada página definía su propio bloque <style> con los mismos
  colores. Si un color se ajustaba en un módulo y no en otro, la app se
  veía inconsistente. Ahora toda página solo necesita:

      from estilos import aplicar_estilos, AZUL, GRANATE, DORADO
      aplicar_estilos()

  y automáticamente hereda paleta, tipografía y el ajuste de modo oscuro.

Modo oscuro:
  Los colores AZUL y GRANATE originales son tonos oscuros pensados para
  fondo blanco. Sobre fondo oscuro pierden contraste y cuesta leerlos.
  Por eso los textos que usan estos colores (títulos, subtítulos,
  encabezados de tarjetas) NO usan el hex fijo directamente: usan
  variables CSS (--azul-texto, --granate-texto) que este módulo
  redefine, más claras, dentro de un bloque
  `@media (prefers-color-scheme: dark)`. Así, cuando el navegador/SO
  del usuario está en modo oscuro, esos textos se aclaran solitos, sin
  tocar nada en Python.

  Los usos de color que NO son texto (borde del sidebar, fondo de
  botones, borde superior de las tarjetas, borde del disclaimer) se
  mantienen con el hex original: son acentos/decoración, no letras, y
  ya tienen buen contraste en ambos modos.
"""

import streamlit as st

# --------------------------------------------------------------------------- #
# Paleta de colores (modo claro) — misma paleta para TODAS las páginas
#   Fondo blanco | Azul oscuro #1F3864 | Granate #800000 | Dorado #C5961A
# --------------------------------------------------------------------------- #
AZUL    = "#1F3864"
GRANATE = "#800000"
DORADO  = "#C5961A"

# Variantes claras usadas SOLO en modo oscuro, para texto (no para fondos)
AZUL_OSCURO_MODO    = "#7FA8D9"   # azul aclarado, legible sobre fondo oscuro
GRANATE_OSCURO_MODO = "#E08A8A"   # granate aclarado, legible sobre fondo oscuro
DORADO_OSCURO_MODO  = "#EAC96B"   # dorado ligeramente más brillante


def aplicar_estilos() -> None:
    """Inyecta el CSS global compartido (paleta, tipografía, tarjetas,
    disclaimer y ajuste automático de modo oscuro).

    Debe llamarse una vez al inicio de cada página, justo después de
    `st.set_page_config(...)`.
    """
    st.markdown(
        f"""
        <style>
            /* Variables de color: claras por defecto, se aclaran más en modo oscuro */
            :root {{
                --azul-texto: {AZUL};
                --granate-texto: {GRANATE};
                --dorado-texto: {DORADO};
            }}
            @media (prefers-color-scheme: dark) {{
                :root {{
                    --azul-texto: {AZUL_OSCURO_MODO};
                    --granate-texto: {GRANATE_OSCURO_MODO};
                    --dorado-texto: {DORADO_OSCURO_MODO};
                }}
            }}

            /* Tipografía */
            html, body, [class*="css"] {{
                font-family: 'Calibri', 'Segoe UI', sans-serif;
            }}

            /* Sidebar (borde decorativo, se mantiene igual en ambos modos) */
            section[data-testid="stSidebar"] {{
                border-right: 3px solid {AZUL};
            }}

            /* Títulos (texto -> usa variable, se aclara en modo oscuro) */
            h1, h2, h3 {{
                color: var(--azul-texto);
            }}

            /* Subtítulo tipo "Markowitz · NSGA-II · ..." (texto granate) */
            .subtitulo-principal {{
                color: var(--granate-texto);
            }}

            /* Botón principal (fondo, se mantiene igual en ambos modos) */
            div.stButton > button {{
                background-color: {AZUL};
                color: #FFFFFF;
                border: 0;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 600;
                width: 100%;
            }}
            div.stButton > button:hover {{
                background-color: {GRANATE};
                color: #FFFFFF;
            }}

            /* Tarjetas de módulos ADAPTABLES */
            .modulo-card {{
                background-color: var(--secondary-background-color);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-top: 4px solid var(--dorado-texto);
                border-radius: 10px;
                padding: 1.1rem 1.3rem;
                height: 100%;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }}
            .modulo-card h4 {{
                color: var(--azul-texto);
                margin: 0 0 0.4rem 0;
            }}
            .modulo-card p {{
                color: var(--text-color);
                font-size: 0.92rem;
                margin: 0;
            }}

            /* Disclaimer ADAPTABLE (Fondo semi-transparente) */
            .disclaimer {{
                background-color: rgba(197, 150, 26, 0.1);
                border-left: 5px solid var(--dorado-texto);
                color: var(--text-color);
                padding: 0.8rem 1rem;
                border-radius: 6px;
                font-size: 0.88rem;
            }}

            /* Renombrar 'app' a 'Dashboard Principal' en el menú lateral */
            div[data-testid="stSidebarNav"] ul li:first-child a span {{
                font-size: 0 !important;
            }}
            div[data-testid="stSidebarNav"] ul li:first-child a span::before {{
                content: "Dashboard Principal" !important;
                font-size: 14px !important;
                font-weight: 500;
                color: var(--text-color);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
