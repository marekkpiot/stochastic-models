import matplotlib.pyplot as plt
import numpy as np

from src.heston import simulate_heston
from src.implied_volatility import (
    implied_volatility_call,
)

initial_stock_price = 100.0
initial_variance = 0.04

rate = 0.03

mean_reversion_speed = 2.0
long_term_variance = 0.04
volatility_of_variance = 0.30
correlation = -0.70

maturity = 1.0

n_steps = 252
n_paths = 20_000

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
    seed=42,
)

terminal_stock_prices = stock_paths[:, -1]

strikes = np.arange(
    70.0,
    135.0,
    5.0,
)

discount_factor = np.exp(
    -rate * maturity
)

heston_prices = []
standard_errors = []
implied_volatilities = []

for strike in strikes:
    payoffs = np.maximum(
        terminal_stock_prices - strike,
        0.0,
    )

    discounted_payoffs = (
        discount_factor * payoffs
    )

    heston_price = np.mean(
        discounted_payoffs
    )

    standard_error = (
        np.std(
            discounted_payoffs,
            ddof=1,
        )
        / np.sqrt(n_paths)
    )

    implied_volatility = (
        implied_volatility_call(
            option_price=heston_price,
            initial_stock_price=(
                initial_stock_price
            ),
            strike=strike,
            rate=rate,
            maturity=maturity,
        )
    )

    heston_prices.append(
        heston_price
    )

    standard_errors.append(
        standard_error
    )

    implied_volatilities.append(
        implied_volatility
    )

print(
    "Strike | Prix Heston | "
    "Erreur standard | Volatilité implicite"
)

print("-" * 65)


for (
    strike,
    price,
    error,
    implied_volatility,
) in zip(
    strikes,
    heston_prices,
    standard_errors,
    implied_volatilities,
):
    print(
        f"{strike:6.1f} | "
        f"{price:11.4f} | "
        f"{error:15.4f} | "
        f"{100 * implied_volatility:8.2f} %"
    )

implied_volatilities = np.asarray(
    implied_volatilities,
)

plt.figure(figsize=(9, 5))

plt.plot(
    strikes,
    100 * implied_volatilities,
    marker="o",
)

plt.axvline(
    initial_stock_price,
    linestyle="--",
    label="Prix initial",
)

plt.title(
    "Volatilité implicite sous le modèle de Heston"
)

plt.xlabel("Strike")
plt.ylabel("Volatilité implicite (%)")
plt.grid()
plt.legend()

plt.savefig(
    "figures/heston_implied_volatility_skew.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
