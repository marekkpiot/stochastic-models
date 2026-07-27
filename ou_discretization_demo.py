import matplotlib.pyplot as plt
import numpy as np

from src.ornstein_uhlenbeck import (
    simulate_ornstein_uhlenbeck,
    simulate_ornstein_uhlenbeck_exact,
)


# Paramètres du processus
initial_value = 3.0
mean_level = 1.0
mean_reversion_speed = 2.0
volatility = 0.40
maturity = 3.0

# On choisit volontairement peu de pas
# afin de rendre l'erreur d'Euler visible.
n_steps = 30
n_paths = 1
seed = 42


times, euler_paths = simulate_ornstein_uhlenbeck(
    initial_value=initial_value,
    mean_level=mean_level,
    mean_reversion_speed=mean_reversion_speed,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=seed,
)


_, exact_paths = simulate_ornstein_uhlenbeck_exact(
    initial_value=initial_value,
    mean_level=mean_level,
    mean_reversion_speed=mean_reversion_speed,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=seed,
)


euler_path = euler_paths[0]
exact_path = exact_paths[0]


maximum_difference = np.max(
    np.abs(euler_path - exact_path)
)

print("Taille du pas :", maturity / n_steps)
print(
    "Écart maximal entre les deux trajectoires :",
    maximum_difference,
)


plt.figure(figsize=(9, 5))

plt.plot(
    times,
    euler_path,
    marker="o",
    label="Euler-Maruyama",
)

plt.plot(
    times,
    exact_path,
    marker="x",
    linestyle="--",
    label="Transition exacte",
)

plt.axhline(
    mean_level,
    linestyle=":",
    label="Niveau moyen",
)

plt.title(
    "Euler-Maruyama et simulation exacte "
    "d'Ornstein-Uhlenbeck"
)
plt.xlabel("Temps")
plt.ylabel("X(t)")
plt.grid()
plt.legend()

plt.savefig(
    "figures/ou_euler_vs_exact.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
