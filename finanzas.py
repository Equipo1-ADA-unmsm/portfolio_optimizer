"""
Funciones de portafolio compartidas
=====================================
Funciones de finanzas cuantitativas (rendimiento esperado, volatilidad,
Sharpe y Sortino) usadas por el módulo de Markowitz y el módulo de
Comparación, que antes las tenían duplicadas con pequeñas variaciones
(una versión con `np.dot`, otra con el operador `@`; algunas sin la
protección `abs()` contra errores de precisión flotante).

Centralizarlas evita que las fórmulas diverjan silenciosamente entre
módulos con el tiempo — algo especialmente importante para que la
Comparación sea, de verdad, una comparación justa entre métodos.

Uso:
    from finanzas import (
        portfolio_performance,
        negative_sharpe_ratio,
        portfolio_volatility,
        calculate_sortino_ratio,
    )
"""

import numpy as np

DIAS_ANIO_DEFAULT = 252


def portfolio_performance(weights, mu, Sigma):
    """Retorno esperado y volatilidad anualizados de un portafolio.

    Acepta `mu`/`Sigma` como arrays de numpy o como Series/DataFrame de
    pandas (ambos son compatibles con las operaciones usadas aquí).

    Returns
    -------
    (retorno, volatilidad) : tuple[float, float]
    """
    p_ret = np.sum(mu * weights)
    # abs() protege contra errores de dominio matemático (raíz de negativo)
    # que pueden aparecer por redondeos de punto flotante.
    p_std = np.sqrt(abs(np.dot(weights.T, np.dot(Sigma, weights))))
    return p_ret, p_std


def negative_sharpe_ratio(weights, mu, Sigma, risk_free_rate=0.0):
    """Sharpe ratio negado (para minimizar con scipy.optimize)."""
    p_ret, p_std = portfolio_performance(weights, mu, Sigma)
    return -(p_ret - risk_free_rate) / p_std if p_std > 0 else 0.0


def portfolio_volatility(weights, mu, Sigma):
    """Solo la volatilidad anualizada (para minimizar mínima varianza)."""
    return portfolio_performance(weights, mu, Sigma)[1]


def calculate_sortino_ratio(weights, historical_returns, risk_free_rate=0.0,
                             dias_anio=DIAS_ANIO_DEFAULT):
    """Sortino ratio de un portafolio a partir de sus retornos históricos."""
    port_returns = historical_returns.dot(weights)
    downside = port_returns[port_returns < 0]
    expected_return = port_returns.mean() * dias_anio
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(dias_anio)
    if downside_std == 0:
        return 0.0
    return (expected_return - risk_free_rate) / downside_std
