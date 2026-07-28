import numpy as np


rng = np.random.default_rng(seed=42)

n_samples = 100_000
rho = -0.7


# Deux suites de normales indépendantes
z_1 = rng.normal(
    loc=0.0,
    scale=1.0,
    size=n_samples,
)

z_2 = rng.normal(
    loc=0.0,
    scale=1.0,
    size=n_samples,
)


# Construction de deux chocs corrélés
price_shocks = z_1

variance_shocks = (
    rho * z_1
    + np.sqrt(1.0 - rho**2) * z_2
)


empirical_correlation = np.corrcoef(
    price_shocks,
    variance_shocks,
)[0, 1]


print("Corrélation demandée :", rho)

print(
    "Corrélation empirique :",
    empirical_correlation,
)