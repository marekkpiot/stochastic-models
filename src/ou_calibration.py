import numpy as np
from scipy.optimize import minimize


def calibrate_ou_naive(
    paths,
    dt: float,
):
    """
    Calibre naïvement les paramètres d'un processus
    d'Ornstein-Uhlenbeck à partir de trajectoires observées.

    On utilise le schéma d'Euler :

        X_{k+1}
        = X_k + theta * (mu - X_k) * dt
          + sigma * sqrt(dt) * Z_k

    theta et mu sont estimés par minimisation d'une MSE.
    sigma est estimé à partir de la dispersion des résidus.
    """

    paths = np.asarray(
        paths,
        dtype=float,
    )

    if dt <= 0:
        raise ValueError(
            "Le pas de temps doit être positif."
        )

    # La fonction accepte aussi une seule trajectoire 1D.
    if paths.ndim == 1:
        paths = paths[np.newaxis, :]

    if paths.ndim != 2:
        raise ValueError(
            "paths doit être un tableau 1D ou 2D."
        )

    if paths.shape[1] < 2:
        raise ValueError(
            "Il faut au moins deux dates."
        )

    # Toutes les valeurs X_k.
    current_values = paths[:, :-1].ravel()

    # Toutes les valeurs X_{k+1}.
    next_values = paths[:, 1:].ravel()

    def objective(parameters):
        """
        Fonction que scipy.optimize doit minimiser.
        """

        mean_reversion_speed = parameters[0]
        mean_level = parameters[1]

        predicted_next_values = (
            current_values
            + mean_reversion_speed
            * (
                mean_level
                - current_values
            )
            * dt
        )

        errors = (
            next_values
            - predicted_next_values
        )

        # La division par sqrt(dt) ne change pas
        # la position du minimum.
        # Elle évite seulement d'avoir des nombres
        # extrêmement petits dans l'optimisation.
        scaled_errors = (
            errors / np.sqrt(dt)
        )

        return np.mean(
            scaled_errors**2
        )

    initial_guess = np.array(
        [
            1.0,
            np.mean(paths),
        ]
    )

    result = minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=[
            (1e-8, None),
            (None, None),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "L'optimisation a échoué : "
            + result.message
        )

    estimated_mean_reversion_speed = (
        result.x[0]
    )

    estimated_mean_level = (
        result.x[1]
    )

    predicted_next_values = (
        current_values
        + estimated_mean_reversion_speed
        * (
            estimated_mean_level
            - current_values
        )
        * dt
    )

    residuals = (
        next_values
        - predicted_next_values
    )

    estimated_volatility = np.sqrt(
        np.mean(residuals**2) / dt
    )

    return {
        "mean_reversion_speed": float(
            estimated_mean_reversion_speed
        ),
        "mean_level": float(
            estimated_mean_level
        ),
        "volatility": float(
            estimated_volatility
        ),
        "mse": float(result.fun),
    }