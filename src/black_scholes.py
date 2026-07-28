import math

from scipy.stats import norm


def black_scholes_call(
    initial_stock_price: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """
    Calcule le prix Black-Scholes d'un call européen
    sans dividende.
    """

    if initial_stock_price <= 0:
        raise ValueError(
            "Le prix initial doit être strictement positif."
        )

    if strike <= 0:
        raise ValueError(
            "Le strike doit être strictement positif."
        )

    if volatility < 0:
        raise ValueError(
            "La volatilité ne peut pas être négative."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être strictement positive."
        )

    # Cas limite : aucune volatilité.
    if volatility == 0:
        return max(
            initial_stock_price
            - strike * math.exp(-rate * maturity),
            0.0,
        )

    square_root_maturity = math.sqrt(maturity)

    d_1 = (
        math.log(initial_stock_price / strike)
        + (
            rate
            + 0.5 * volatility**2
        )
        * maturity
    ) / (
        volatility * square_root_maturity
    )

    d_2 = (
        d_1
        - volatility * square_root_maturity
    )

    call_price = (
        initial_stock_price * norm.cdf(d_1)
        - strike
        * math.exp(-rate * maturity)
        * norm.cdf(d_2)
    )

    return float(call_price)