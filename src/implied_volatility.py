import math

import numpy as np
from scipy.optimize import brentq

from src.black_scholes import black_scholes_call


def implied_volatility_call(
    option_price: float,
    initial_stock_price: float,
    strike: float,
    rate: float,
    maturity: float,
) -> float:
    """
    Trouve la volatilité Black-Scholes qui reproduit
    le prix fourni pour un call européen.

    Retourne np.nan si aucun résultat cohérent
    ne peut être trouvé.
    """

    discounted_strike = (
        strike * math.exp(-rate * maturity)
    )

    lower_bound = max(
        initial_stock_price - discounted_strike,
        0.0,
    )

    upper_bound = initial_stock_price

    # Un prix de call cohérent doit se trouver
    # entre ces deux bornes.
    if not lower_bound <= option_price < upper_bound:
        return np.nan

    def pricing_difference(
        tested_volatility: float,
    ) -> float:
        black_scholes_price = black_scholes_call(
            initial_stock_price=initial_stock_price,
            strike=strike,
            rate=rate,
            volatility=tested_volatility,
            maturity=maturity,
        )

        return (
            black_scholes_price
            - option_price
        )

    try:
        implied_volatility = brentq(
            pricing_difference,
            1e-8,
            5.0,
        )

    except ValueError:
        return np.nan

    return float(implied_volatility)