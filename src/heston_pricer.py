import numpy as np

from src.heston import simulate_heston


def price_european_option_heston(
    initial_stock_price: float,
    strike: float,
    initial_variance: float,
    rate: float,
    mean_reversion_speed: float,
    long_term_variance: float,
    volatility_of_variance: float,
    correlation: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "call",
    seed: int | None = None,
):
    """
    Calcule le prix d'une option européenne sous le modèle
    de Heston par la méthode de Monte-Carlo.

    Returns
    -------
    estimated_price:
        Estimation Monte-Carlo du prix de l'option.

    standard_error:
        Erreur standard de l'estimation.
    """

    if strike <= 0:
        raise ValueError(
            "Le prix d'exercice doit être strictement positif."
        )

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type doit être égal à 'call' ou 'put'."
        )

    _, stock_paths, _ = simulate_heston(
        initial_stock_price=initial_stock_price,
        initial_variance=initial_variance,
        rate=rate,
        mean_reversion_speed=mean_reversion_speed,
        long_term_variance=long_term_variance,
        volatility_of_variance=volatility_of_variance,
        correlation=correlation,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    terminal_stock_prices = stock_paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(
            terminal_stock_prices - strike,
            0.0,
        )
    else:
        payoffs = np.maximum(
            strike - terminal_stock_prices,
            0.0,
        )

    discount_factor = np.exp(
        -rate * maturity
    )

    discounted_payoffs = (
        discount_factor * payoffs
    )

    estimated_price = np.mean(
        discounted_payoffs
    )

    standard_error = (
        np.std(
            discounted_payoffs,
            ddof=1,
        )
        / np.sqrt(n_paths)
    )

    return float(estimated_price), float(standard_error)