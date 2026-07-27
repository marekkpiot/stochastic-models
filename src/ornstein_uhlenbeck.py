import numpy as np


def simulate_ornstein_uhlenbeck(
    initial_value: float,
    mean_level: float,
    mean_reversion_speed: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
):
    """
    Simule des trajectoires d'un processus d'Ornstein-Uhlenbeck
    avec le schéma d'Euler-Maruyama.

    L'équation est :

        dX_t = theta * (mu - X_t) dt + sigma dW_t

    Parameters
    ----------
    initial_value:
        Valeur initiale X_0.

    mean_level:
        Niveau moyen de long terme mu.

    mean_reversion_speed:
        Vitesse de retour à la moyenne theta.

    volatility:
        Intensité du bruit sigma.

    maturity:
        Durée totale de la simulation.

    n_steps:
        Nombre de pas de temps.

    n_paths:
        Nombre de trajectoires simulées.

    seed:
        Graine aléatoire.

    Returns
    -------
    times:
        Tableau contenant les dates.

    paths:
        Tableau contenant les trajectoires simulées.
    """

    if mean_reversion_speed <= 0:
        raise ValueError(
            "La vitesse de retour à la moyenne doit être positive."
        )

    if volatility < 0:
        raise ValueError(
            "La volatilité ne peut pas être négative."
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

    normal_shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_paths, n_steps),
    )

    paths = np.zeros(
        (n_paths, n_steps + 1)
    )

    paths[:, 0] = initial_value

    for step in range(n_steps):
        current_values = paths[:, step]

        drift = (
            mean_reversion_speed
            * (mean_level - current_values)
            * dt
        )

        diffusion = (
            volatility
            * np.sqrt(dt)
            * normal_shocks[:, step]
        )

        paths[:, step + 1] = (
            current_values
            + drift
            + diffusion
        )

    times = np.linspace(
        0.0,
        maturity,
        n_steps + 1,
    )

    return times, paths