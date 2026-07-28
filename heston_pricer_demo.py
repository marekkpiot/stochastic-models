from src.heston_pricer import (
    price_european_option_heston,
)


initial_stock_price = 100.0
strike = 100.0

initial_variance = 0.04
rate = 0.03

mean_reversion_speed = 2.0
long_term_variance = 0.04
volatility_of_variance = 0.30
correlation = -0.70

maturity = 1.0

n_steps = 252
n_paths = 10_000


call_price, call_standard_error = (
    price_european_option_heston(
        initial_stock_price=initial_stock_price,
        strike=strike,
        initial_variance=initial_variance,
        rate=rate,
        mean_reversion_speed=mean_reversion_speed,
        long_term_variance=long_term_variance,
        volatility_of_variance=volatility_of_variance,
        correlation=correlation,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        option_type="call",
        seed=42,
    )
)


put_price, put_standard_error = (
    price_european_option_heston(
        initial_stock_price=initial_stock_price,
        strike=strike,
        initial_variance=initial_variance,
        rate=rate,
        mean_reversion_speed=mean_reversion_speed,
        long_term_variance=long_term_variance,
        volatility_of_variance=volatility_of_variance,
        correlation=correlation,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        option_type="put",
        seed=42,
    )
)


print("Prix du call sous Heston")
print(f"Prix estimé : {call_price:.4f}")
print(
    f"Erreur standard : "
    f"{call_standard_error:.4f}"
)

print()

print("Prix du put sous Heston")
print(f"Prix estimé : {put_price:.4f}")
print(
    f"Erreur standard : "
    f"{put_standard_error:.4f}"
)

call_lower_bound = (
    call_price
    - 1.96 * call_standard_error
)

call_upper_bound = (
    call_price
    + 1.96 * call_standard_error
)

print(
    "Intervalle de confiance à 95 % : "
    f"[{call_lower_bound:.4f}, "
    f"{call_upper_bound:.4f}]"
)

import numpy as np


left_side = call_price - put_price

right_side = (
    initial_stock_price
    - strike * np.exp(-rate * maturity)
)

print()
print("Vérification de la parité call-put")

print(
    f"Call - Put : {left_side:.4f}"
)

print(
    f"S0 - K exp(-rT) : {right_side:.4f}"
)

print(
    f"Écart : "
    f"{abs(left_side - right_side):.4f}"
)

print()
print("Effet de la corrélation")


for tested_correlation in [-0.7, 0.0, 0.7]:
    price, error = price_european_option_heston(
        initial_stock_price=initial_stock_price,
        strike=strike,
        initial_variance=initial_variance,
        rate=rate,
        mean_reversion_speed=mean_reversion_speed,
        long_term_variance=long_term_variance,
        volatility_of_variance=volatility_of_variance,
        correlation=tested_correlation,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        option_type="call",
        seed=42,
    )

    print(
        f"rho = {tested_correlation:+.1f} "
        f"→ prix = {price:.4f} "
        f"± {error:.4f}"
    )