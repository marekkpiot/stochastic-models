import matplotlib.pyplot as plt
import numpy as np

from src.heston import simulate_heston

initial_stock_price = 100.0
initial_variance = 0.04

rate = 0.03

mean_reversion_speed = 2.0
long_term_variance = 0.04
volatility_of_variance = 0.30

correlation = -0.70

maturity = 1.0
n_steps = 252
n_paths = 10


times, stock_paths, variance_paths = simulate_heston(
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
    seed=42,
)


print("Forme des trajectoires de prix :", stock_paths.shape)

print(
    "Forme des trajectoires de variance :",
    variance_paths.shape,
)

print()

print(
    "Prix final de la première trajectoire :",
    stock_paths[0, -1],
)

print(
    "Variance finale de la première trajectoire :",
    variance_paths[0, -1],
)

print(
    "Volatilité finale de la première trajectoire :",
    np.sqrt(variance_paths[0, -1]),
)

plt.figure(figsize=(9, 5))

for path in stock_paths:
    plt.plot(times, path)

plt.axhline(
    initial_stock_price,
    linestyle="--",
    label="Prix initial",
)

plt.title("Trajectoires du prix dans le modèle de Heston")
plt.xlabel("Temps")
plt.ylabel("Prix de l'actif")
plt.grid()
plt.legend()

plt.savefig(
    "figures/heston_stock_paths.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

plt.figure(figsize=(9, 5))

for path in variance_paths:
    plt.plot(times, path)

plt.axhline(
    long_term_variance,
    linestyle="--",
    label="Variance de long terme",
)

plt.title("Trajectoires de variance dans le modèle de Heston")
plt.xlabel("Temps")
plt.ylabel("Variance")
plt.grid()
plt.legend()

plt.savefig(
    "figures/heston_variance_paths.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()