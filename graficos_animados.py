"""
Animación de "reproducción" para gráficas de Plotly
=====================================================
Helper compartido para las gráficas de evolución de riqueza y el timeline
de rebalanceos (módulos 1-4). A diferencia de la frontera eficiente, el
frente de Pareto (NSGA-II) o el backward induction de Bellman (DP) — donde
sí hay un cómputo real y lento que vale la pena animar MIENTRAS ocurre—,
estas gráficas se calculan a partir de una simulación vectorizada o un loop
día a día que ya toma milisegundos. No hay "construcción en vivo" real que
mostrar aquí.

Por eso este helper no anima un cómputo en curso: agrega frames NATIVOS de
Plotly (`fig.frames`) más un botón ▶ Reproducir / ⏸ Pausa y un slider, para
que el usuario vea la serie "construirse" a lo largo del tiempo como una
REPRODUCCIÓN. La animación corre enteramente en el navegador (JavaScript),
sin que Streamlit tenga que seguir empujando actualizaciones desde el
backend — a diferencia de las animaciones de los otros 3 módulos, aquí no
hace falta ningún manejo de caché especial: el resultado (con sus frames ya
incluidos) se cachea como cualquier otra figura, y el usuario decide cuándo
darle "Reproducir".

Por qué vive en su propio archivo (mismo criterio que estilos.py,
sidebar.py, datos.py y finanzas.py): esta misma necesidad se repite en las
4 páginas (evolución de riqueza en los 4 módulos, timeline de rebalanceos
en el módulo de DP), así que centralizarla evita reimplementar la
construcción de frames 5 veces con pequeñas variaciones.

Uso:
    from graficos_animados import agregar_animacion_reveal

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=riqueza_a, name="Estrategia A"))
    fig.add_trace(go.Scatter(x=fechas, y=riqueza_b, name="Estrategia B"))
    fig = agregar_animacion_reveal(fig)
    st.plotly_chart(fig, width='stretch')
"""

import bisect

import numpy as np
import plotly.graph_objects as go


def agregar_animacion_reveal(fig, n_pasos=60, duracion_ms=40):
    """Agrega frames de reproducción a TODAS las trazas de `fig`,
    revelando sus puntos progresivamente según su eje x (asumido temporal
    y ordenado ascendentemente).

    Los `shapes` de la figura (como una línea de referencia agregada con
    `fig.add_hline(...)`) NO se animan: permanecen fijos en cada frame,
    que es el comportamiento esperado para, por ejemplo, la línea del
    capital inicial.

    Parameters
    ----------
    fig : go.Figure
        Figura ya armada con sus trazas COMPLETAS. Esos datos completos se
        usan para construir los frames; el estado inicial de la figura se
        reemplaza por el primer frame (un recorte pequeño), para que
        "Reproducir" arranque desde el principio de la serie en vez de
        mostrar todo ya dibujado desde el primer instante.
    n_pasos : int
        Cantidad de frames a generar. ~60 da una reproducción fluida de
        2-3 segundos sin sobrecargar al navegador con demasiados frames.
    duracion_ms : int
        Duración de cada frame (ms) durante la reproducción automática.

    Returns
    -------
    go.Figure
        La misma figura, con `.frames`, botones y slider ya agregados.
    """
    if not fig.data:
        return fig

    trazas_x = [list(tr.x) if tr.x is not None else [] for tr in fig.data]
    trazas_y = [list(tr.y) if tr.y is not None else [] for tr in fig.data]
    # `text` (p. ej. el número de periodo en el timeline de rebalanceos del
    # módulo DP) debe truncarse EN SINCRONÍA con x/y: si una traza tiene
    # texto por punto y no lo truncamos igual, Plotly conserva el arreglo
    # de texto completo mientras x/y quedan recortados, desalineando las
    # etiquetas de los puntos que sí se muestran en cada frame.
    trazas_text = [
        list(tr.text) if getattr(tr, "text", None) is not None else None
        for tr in fig.data
    ]

    longitudes = [len(xs) for xs in trazas_x]
    if max(longitudes, default=0) < 2:
        return fig  # series demasiado cortas: nada que animar

    # La traza con más puntos (la serie diaria más larga) define la
    # resolución temporal real de los cortes a animar.
    idx_larga = max(range(len(trazas_x)), key=lambda i: longitudes[i])
    x_ref = trazas_x[idx_larga]

    cortes_pos = sorted(set(np.linspace(0, len(x_ref) - 1, n_pasos, dtype=int).tolist()))
    frames = []
    for pos in cortes_pos:
        corte = x_ref[pos]
        data_frame = []
        for xs, ys, texts in zip(trazas_x, trazas_y, trazas_text):
            # bisect en vez de un loop con sum(): cada traza es una serie
            # temporal ordenada, así que basta ubicar dónde cae `corte`.
            m = bisect.bisect_right(xs, corte)
            kwargs = dict(x=xs[:m], y=ys[:m])
            if texts is not None:
                kwargs["text"] = texts[:m]
            data_frame.append(go.Scatter(**kwargs))
        frames.append(go.Frame(data=data_frame, name=str(pos)))

    fig.frames = frames

    # Estado inicial = primer frame, para que la reproducción arranque
    # "vacía" y se vea construirse — no con la serie completa ya trazada.
    primer = frames[0]
    for tr, d in zip(fig.data, primer.data):
        tr.x = d.x
        tr.y = d.y
        if d.text is not None:
            tr.text = d.text

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=1.12, xanchor="left", yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(label="▶ Reproducir", method="animate",
                     args=[None, dict(frame=dict(duration=duracion_ms, redraw=True),
                                       fromcurrent=True,
                                       transition=dict(duration=0))]),
                dict(label="⏸ Pausa", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.15, y=1.12, len=0.85,
            xanchor="left", yanchor="top",
            pad=dict(t=0),
            currentvalue=dict(visible=False),
            steps=[
                dict(method="animate", label="",
                     args=[[fr.name], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=True),
                                            transition=dict(duration=0))])
                for fr in frames
            ],
        )],
    )
    return fig
