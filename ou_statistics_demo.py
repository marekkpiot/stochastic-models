import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

from src.ornstein_uhlenbeck import (
    simulate_ornstein_uhlenbeck,
)

# Paramètres du processus
initial_value = 3.0
mean_level = 1.0
mean_reversion_speed = 2.0
volatility = 0.40

# Paramètres numériques
maturity = 3.0
n_steps = 3 * 252
n_paths = 10_000


times, paths = simulate_ornstein_uhlenbeck(
    initial_value=initial_value,
    mean_level=mean_level,
    mean_reversion_speed=mean_reversion_speed,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=42,
)

empirical_mean = np.mean(
    paths,
    axis=0,
)

theoretical_mean = (
    mean_level
    + (initial_value - mean_level)
    * np.exp(-mean_reversion_speed * times)
)

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    empirical_mean,
    label="Moyenne empirique",
)

plt.plot(
    times,
    theoretical_mean,
    linestyle="--",
    label="Moyenne théorique",
)

plt.axhline(
    mean_level,
    linestyle=":",
    label="Niveau moyen mu",
)

plt.title("Moyenne du processus d'Ornstein-Uhlenbeck")
plt.xlabel("Temps")
plt.ylabel("Moyenne de X(t)")
plt.grid()
plt.legend()

plt.savefig(
    "figures/ou_mean_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

empirical_variance = np.var(
    paths,
    axis=0,
)

theoretical_variance = (
    volatility**2
    / (2.0 * mean_reversion_speed)
    * (
        1.0
        - np.exp(
            -2.0
            * mean_reversion_speed
            * times
        )
    )
)

stationary_variance = (
    volatility**2
    / (2.0 * mean_reversion_speed)
)

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    empirical_variance,
    label="Variance empirique",
)

plt.plot(
    times,
    theoretical_variance,
    linestyle="--",
    label="Variance théorique",
)

plt.axhline(
    stationary_variance,
    linestyle=":",
    label="Variance stationnaire",
)

plt.title("Variance du processus d'Ornstein-Uhlenbeck")
plt.xlabel("Temps")
plt.ylabel("Variance de X(t)")
plt.grid()
plt.legend()

plt.savefig(
    "figures/ou_variance_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

terminal_values = paths[:, -1]

terminal_mean = np.mean(terminal_values)
terminal_variance = np.var(terminal_values)

stationary_mean = mean_level

stationary_standard_deviation = np.sqrt(
    stationary_variance
)

print("Statistiques à la date finale")
print(f"Moyenne empirique : {terminal_mean:.4f}")
print(f"Moyenne attendue à long terme : {stationary_mean:.4f}")

print()

print(f"Variance empirique : {terminal_variance:.4f}")
print(
    "Variance stationnaire théorique : "
    f"{stationary_variance:.4f}"
)

print()

print(
    "Écart-type stationnaire théorique : "
    f"{stationary_standard_deviation:.4f}"
)

plt.figure(figsize=(9, 5))

plt.hist(
    terminal_values,
    bins=50,
    density=True,
    alpha=0.6,
    label="Distribution empirique",
)

x_values = np.linspace(
    mean_level - 4 * stationary_standard_deviation,
    mean_level + 4 * stationary_standard_deviation,
    500,
)

theoretical_density = norm.pdf(
    x_values,
    loc=mean_level,
    scale=stationary_standard_deviation,
)

plt.plot(
    x_values,
    theoretical_density,
    linewidth=2,
    label="Densité stationnaire théorique",
)

plt.axvline(
    mean_level,
    linestyle="--",
    label="Niveau moyen mu",
)

plt.title(
    "Distribution stationnaire du processus "
    "d'Ornstein-Uhlenbeck"
)
plt.xlabel("Valeur de X(T)")
plt.ylabel("Densité")
plt.grid()
plt.legend()

plt.savefig(
    "figures/ou_stationary_distribution.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
