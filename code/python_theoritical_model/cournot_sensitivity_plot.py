# Revised simulation with an outside option (reservation price) and Poisson N2 truncated to [0, 9]
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
kappa_fixed = 0.9
N1_fixed = 3
reservation_price = 0.10   # outside option (price floor)
gamma_values = [0, 0.5, 2, 4, 7]
mu_range = np.arange(1, 9, 1)   # expected N2 from 1 to 8
n_samples = 5000
s_grid = np.linspace(0, 1, 25)  # s-grid resolution = 25 points

# -----------------------------
# Utility and helpers
# -----------------------------
def crra_utility(pi, gamma):
    with np.errstate(divide='ignore', invalid='ignore'):
        if gamma == 1:
            return np.log(pi)
        else:
            return (np.power(pi, 1 - gamma) - 1) / (1 - gamma)

def effective_price(N, p_floor):
    """Cournot price with outside option floor."""
    return np.maximum(N / (N + 1), p_floor)

def optimal_s(N2_samples, gamma, kappa, p_floor):
    prices2 = effective_price(N2_samples, p_floor)
    expected_utils = []
    for s in s_grid:
        income1 = (1 - s) * (N1_fixed / (N1_fixed + 1))
        income2 = s * kappa * prices2
        income_total = income1 + income2
        EU = np.mean(crra_utility(income_total, gamma))
        expected_utils.append(EU)
    return s_grid[np.argmax(expected_utils)]

def closed_form_s(N2_samples, kappa, p_floor):
    income1 = N1_fixed / (N1_fixed + 1)
    income2 = kappa * np.mean(effective_price(N2_samples, p_floor))
    return 1.0 if income2 > income1 else 0.0

# -----------------------------
# Simulation
# -----------------------------
s_star_by_mu_gamma = {gamma: [] for gamma in gamma_values}

rng = np.random.default_rng(42)
for mu in mu_range:
    # Draw N2 ~ Poisson(mu) and truncate to [0, 9]
    N2_samples = rng.poisson(mu, n_samples)
    N2_samples = np.clip(N2_samples, 0, 9)

    for gamma in gamma_values:
        if gamma == 0:
            s_star = closed_form_s(N2_samples, kappa_fixed, reservation_price)
        else:
            s_star = optimal_s(N2_samples, gamma, kappa_fixed, reservation_price)
        s_star_by_mu_gamma[gamma].append(s_star)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Blues colormap that darkens with higher gamma; keep gamma=0 not too light
blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(gamma_values)))

for idx, gamma in enumerate(gamma_values):
    ax.plot(mu_range, s_star_by_mu_gamma[gamma], label=f"$\\gamma$ = {gamma}", color=blues[idx], linewidth=2)

# Vertical dashed line at E[N2] = N1
ax.axvline(x=N1_fixed, color='gray', linestyle='--', linewidth=1)
ax.text(N1_fixed + 0.1, 0.05, "$E[N_2] = N_1$", color='gray')

ax.set_xlabel("Expected $N_2$")
ax.set_ylabel("Optimal Storage Share $s^*$")
ax.set_title("Optimal Storage vs Expected Second-Period Buyer Count\n(Cournot with Outside Option)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(title="Risk Aversion")
plt.tight_layout()


# Save the figure
plt.savefig("buyer_count_sensitivity_cournot.png", dpi=300, bbox_inches="tight")


plt.show()
