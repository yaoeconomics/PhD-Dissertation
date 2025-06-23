# Re-import libraries due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Re-define utility and optimization functions
def crra_utility(pi, gamma):
    with np.errstate(divide='ignore', invalid='ignore'):
        if gamma == 1:
            return np.log(pi)
        else:
            return (np.power(pi, 1 - gamma) - 1) / (1 - gamma)

def optimal_s(N2_samples, gamma, kappa):
    s_grid = np.linspace(0, 1, 25)
    expected_utils = []
    for s in s_grid:
        income1 = (1 - s) * 3 / 4  # N1 = 3
        prices2 = N2_samples / (N2_samples + 1)
        income2 = s * kappa * prices2
        income_total = income1 + income2
        EU = np.mean(crra_utility(income_total, gamma))
        expected_utils.append(EU)
    return s_grid[np.argmax(expected_utils)]

def closed_form_s(N2_samples, kappa):
    income1 = 0.75
    income2 = kappa * np.mean(N2_samples / (N2_samples + 1))
    return 1.0 if income2 > income1 else 0.0

# Simulation parameters
kappa_fixed = 0.9
N1_fixed = 3
gamma_values = [0, 0.5, 2, 4, 7]
mu_range = np.arange(1, 9, 1)
n_samples = 5000

# Container for results
s_star_by_mu_gamma = {gamma: [] for gamma in gamma_values}

# Compute for each mu and gamma
for mu in mu_range:
    N2_samples = np.random.poisson(mu, n_samples)
    N2_samples = np.clip(N2_samples, 0, 9)

    for gamma in gamma_values:
        if gamma == 0:
            s_star = closed_form_s(N2_samples, kappa_fixed)
        else:
            s_star = optimal_s(N2_samples, gamma, kappa_fixed)
        s_star_by_mu_gamma[gamma].append(s_star)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(gamma_values)))

for idx, gamma in enumerate(gamma_values):
    ax.plot(mu_range, s_star_by_mu_gamma[gamma], label=f"$\\gamma$ = {gamma}", color=blues[idx], linewidth=2)

# Reference line
ax.axvline(x=N1_fixed, color='gray', linestyle='--', linewidth=1)
ax.text(N1_fixed + 0.1, 0.05, "$E[N_2] = N_1$", color='gray')

# Final touches
ax.set_xlabel("Expected $N_2$")
ax.set_ylabel("Optimal Storage Share $s^*$")
ax.set_title("Optimal Storage vs Expected Second-Period Buyer Count")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(title="Risk Aversion")
plt.tight_layout()
plt.show()
