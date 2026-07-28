from src.ornstein_uhlenbeck import (
    simulate_ornstein_uhlenbeck,
)

from src.ou_calibration import (
    calibrate_ou_naive,
)


# Paramètres réels utilisés pour générer les données.
true_initial_value = 3.0
true_mean_level = 1.0
true_mean_reversion_speed = 1.5
true_volatility = 0.30


# On génère plusieurs trajectoires synthétiques.
maturity = 5.0
n_steps = 5 * 252
n_paths = 100

dt = maturity / n_steps


_, synthetic_paths = (
    simulate_ornstein_uhlenbeck(
        initial_value=true_initial_value,
        mean_level=true_mean_level,
        mean_reversion_speed=(
            true_mean_reversion_speed
        ),
        volatility=true_volatility,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=42,
    )
)


estimated_parameters = calibrate_ou_naive(
    paths=synthetic_paths,
    dt=dt,
)


print("Calibration naïve du processus OU")
print()

print(
    "Paramètre                  "
    "Vraie valeur     Valeur estimée"
)

print("-" * 55)

print(
    f"Vitesse de retour theta    "
    f"{true_mean_reversion_speed:10.4f}     "
    f"{estimated_parameters['mean_reversion_speed']:10.4f}"
)

print(
    f"Niveau moyen mu            "
    f"{true_mean_level:10.4f}     "
    f"{estimated_parameters['mean_level']:10.4f}"
)

print(
    f"Volatilité sigma           "
    f"{true_volatility:10.4f}     "
    f"{estimated_parameters['volatility']:10.4f}"
)

print()

print(
    "MSE finale :",
    estimated_parameters["mse"],
)