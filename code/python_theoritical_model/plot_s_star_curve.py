
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters
theta1 = 0.5
p1 = 1 / (1 + theta1)
kappa = 0.9
delta = 1.0
num_draws = 5000
variance = 0.02
mu_values = np.arange(0.05, 0.951, 0.01)
gamma_values = [0, 0.5, 2, 4, 7]
s_grid = np.linspace(0, 1, 25)

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_s(theta2_draws, gamma):
    if gamma == 0:
        expected_p2 = np.mean(1 / (1 + theta2_draws))
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0
    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa / (1 + theta2_draws)
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)
    return s_grid[np.argmax(utilities)]

def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

results = {gamma: [] for gamma in gamma_values}

for mu in mu_values:
    alpha, beta_param = compute_beta_params(mu, variance)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)
    for gamma in gamma_values:
        s_star = optimize_s(theta2_draws, gamma)
        results[gamma].append(s_star)

colors = plt.cm.Blues(np.linspace(0.3, 1, len(gamma_values)))
plt.figure(figsize=(10, 6))

for i, gamma in enumerate(gamma_values):
    plt.plot(mu_values, results[gamma], label=f"$\\gamma = {gamma}$", color=colors[i], linewidth=2)

plt.axvline(theta1, color='k', linestyle='--', label="$\\mathbb{E}[\\theta_2] = \\theta_1$")
plt.xlabel("Expected second-period buyer power $\\mathbb{E}[\\theta_2]$")
plt.ylabel("Optimal storage share $s^*$")
plt.title("Optimal Storage Share $s^*$ vs. Expected Buyer Power")
plt.legend(title="Risk Aversion $\\gamma$")
plt.grid(True)
plt.tight_layout()
plt.show()


# Re-import after code execution state reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Constants ---
theta1 = 0.5
p1 = 1 / (1 + theta1)
kappa = 0.9
delta = 1.0
num_draws = 5000
variance = 0.02
s_grid = np.linspace(0, 1, 25)

# CRRA utility function
def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

# Optimizer
def optimize_s(theta2_draws, gamma):
    if gamma == 0:
        expected_p2 = np.mean(1 / (1 + theta2_draws))
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0
    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa / (1 + theta2_draws)
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)
    return s_grid[np.argmax(utilities)]

# Compute beta parameters from mean and variance
def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

# --- Main Simulation ---
mu_values_focus = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
gamma_grid = np.linspace(0, 10, 100)
results_focus = {mu: [] for mu in mu_values_focus}

for mu in mu_values_focus:
    alpha, beta_param = compute_beta_params(mu, variance)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)
    for gamma in gamma_grid:
        s_star = optimize_s(theta2_draws, gamma)
        results_focus[mu].append(s_star)

# --- Plotting ---
colors = plt.cm.Blues(np.linspace(0.3, 1, len(mu_values_focus)))
plt.figure(figsize=(10, 6))

for i, mu in enumerate(mu_values_focus):
    plt.plot(gamma_grid, results_focus[mu], label=f"$\\mu = {mu}$", color=colors[i], linewidth=2)

plt.xlabel("Risk aversion $\\gamma$")
plt.ylabel("Optimal storage share $s^*$")
plt.title("Impact of Beliefs ($\\mu$) on Storage: $s^*$ vs. $\\gamma$")
plt.legend(title="Mean of $\\theta_2$")
plt.grid(True)
plt.tight_layout()
plt.show()
