import numpy as np


def simulate_heston(
    initial_stock_price: float,
    initial_variance: float,
    rate: float,
    mean_reversion_speed: float,
    long_term_variance: float,
    volatility_of_variance: float,
    correlation: float,
    maturity: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
):
    """
    Simule des trajectoires du modèle de Heston.

    Le modèle est :

        dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S

        dv_t = kappa (theta - v_t) dt
               + xi sqrt(v_t) dW_t^v

    Les deux mouvements browniens ont une corrélation rho.
    """

    if initial_stock_price <= 0:
        raise ValueError(
            "Le prix initial doit être strictement positif."
        )

    if initial_variance < 0:
        raise ValueError(
            "La variance initiale ne peut pas être négative."
        )

    if mean_reversion_speed <= 0:
        raise ValueError(
            "La vitesse de retour à la moyenne doit être positive."
        )

    if long_term_variance < 0:
        raise ValueError(
            "La variance de long terme ne peut pas être négative."
        )

    if volatility_of_variance < 0:
        raise ValueError(
            "La volatilité de la variance ne peut pas être négative."
        )

    if not -1.0 <= correlation <= 1.0:
        raise ValueError(
            "La corrélation doit être comprise entre -1 et 1."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être positive."
        )

    if n_steps <= 0:
        raise ValueError(
            "Le nombre de pas doit être positif."
        )

    if n_paths <= 0:
        raise ValueError(
            "Le nombre de trajectoires doit être positif."
        )

    rng = np.random.default_rng(seed)

    dt = maturity / n_steps

    # Deux suites de variables normales indépendantes.
    z_1 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_paths, n_steps),
    )

    z_2 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_paths, n_steps),
    )

    # Choc utilisé pour le prix.
    stock_shocks = z_1

    # Choc utilisé pour la variance,
    # corrélé avec le choc du prix.
    variance_shocks = (
        correlation * z_1
        + np.sqrt(1.0 - correlation**2) * z_2
    )

    stock_paths = np.zeros(
        (n_paths, n_steps + 1)
    )

    variance_paths = np.zeros(
        (n_paths, n_steps + 1)
    )

    stock_paths[:, 0] = initial_stock_price
    variance_paths[:, 0] = initial_variance

    for step in range(n_steps):
        current_stock_prices = stock_paths[:, step]

        # La variance utilisée dans les calculs
        # est toujours positive ou nulle.
        current_variances = np.maximum(
            variance_paths[:, step],
            0.0,
        )

        variance_drift = (
            mean_reversion_speed
            * (
                long_term_variance
                - current_variances
            )
            * dt
        )

        variance_diffusion = (
            volatility_of_variance
            * np.sqrt(
                current_variances * dt
            )
            * variance_shocks[:, step]
        )

        next_variances = (
            current_variances
            + variance_drift
            + variance_diffusion
        )

        # On empêche la variance simulée
        # de devenir négative.
        variance_paths[:, step + 1] = np.maximum(
            next_variances,
            0.0,
        )

        stock_drift = (
            rate
            - 0.5 * current_variances
        ) * dt

        stock_diffusion = (
            np.sqrt(
                current_variances * dt
            )
            * stock_shocks[:, step]
        )

        stock_paths[:, step + 1] = (
            current_stock_prices
            * np.exp(
                stock_drift
                + stock_diffusion
            )
        )

    times = np.linspace(
        0.0,
        maturity,
        n_steps + 1,
    )

    return times, stock_paths, variance_paths