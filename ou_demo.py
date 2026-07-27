import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.ornstein_uhlenbeck import (
    simulate_ornstein_uhlenbeck,
)



initial_value = 3.0
mean_level = 1.0
mean_reversion_speed = 2.0
volatility = 0.40

maturity = 3.0
n_steps = 3 * 252
n_paths = 10


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


print("Forme des dates :", times.shape)
print("Forme des trajectoires :", paths.shape)

print()
print("Valeur initiale :", paths[0, 0])
print("Valeur finale de la première trajectoire :", paths[0, -1])


plt.figure(figsize=(9, 5))

for path in paths:
    plt.plot(times, path)

plt.axhline(
    mean_level,
    linestyle="--",
    label="Niveau moyen",
)

plt.title("Processus d'Ornstein-Uhlenbeck")
plt.xlabel("Temps")
plt.ylabel("X(t)")
plt.grid()
plt.legend()

plt.savefig(
    "figures/ou_paths.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()