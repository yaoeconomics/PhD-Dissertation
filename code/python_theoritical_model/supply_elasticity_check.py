import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --------------------------------
# Paths
# --------------------------------
# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上一级目录
parent_dir = os.path.dirname(current_dir)
# 上两级目录
grandparent_dir = os.path.dirname(parent_dir)
# 上两级平行目录（例如 results）
target_dir = os.path.join(grandparent_dir, "model_figures")

# 如果目标文件夹不存在就新建
os.makedirs(target_dir, exist_ok=True)

# --------------------------------
# Global font settings (larger)
# --------------------------------
plt.rcParams.update({
    "axes.titlesize": 22,      # panel titles
    "axes.labelsize": 20,      # axis labels
    "xtick.labelsize": 18,     # x tick labels
    "ytick.labelsize": 18,     # y tick labels
    "legend.fontsize": 18,     # legend entries
    "legend.title_fontsize": 19,
    "figure.titlesize": 26     # suptitle
})

# --------------------------------
# Constants
# --------------------------------
theta1 = 0.5
# First-period price: normalized ε1 = 1
p1 = 1 / (1 + theta1)

kappa = 0.9
delta = 1.0
num_draws = 5000
variance = 0.05          # Var(theta_2)
s_grid = np.linspace(0, 1, 100)

# Set of second-period supply elasticities
eps2_values = [0.75, 1.00, 1.25]

# --------------------------------
# Helper functions
# --------------------------------
def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_s(theta2_draws, gamma, eps2):
    """
    θ2 ~ Beta(μ, variance)
    p2(θ2, ε2) = ε2 / (ε2 + θ2)
    """
    # Second-period price draws given ε2
    p2_draws = eps2 / (eps2 + theta2_draws)

    if gamma == 0:
        expected_p2 = np.mean(p2_draws)
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0

    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa * p2_draws
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)
    return s_grid[np.argmax(utilities)]

def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

# --------------------------------
# Main Simulation
# --------------------------------
mu_fixed = 0.2
gamma_grid = np.linspace(0, 10, 100)
results_eps2 = {eps2: [] for eps2 in eps2_values}
expected_p2 = {}

# Draw θ2 once (same uncertainty for all ε2)
alpha, beta_param = compute_beta_params(mu_fixed, variance)
theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)

for eps2 in eps2_values:
    # implied expected second-period price for this ε2
    p2_draws = eps2 / (eps2 + theta2_draws)
    expected_p2[eps2] = np.mean(p2_draws)

    for gamma in gamma_grid:
        s_star = optimize_s(theta2_draws, gamma, eps2)
        results_eps2[eps2].append(s_star)

# Optional: print implied E[p2]
for eps2 in eps2_values:
    print(f"eps2 = {eps2:.2f}, E[p2] ≈ {expected_p2[eps2]:.4f}")

# --------------------------------
# Plotting: 2-panel figure
# --------------------------------
colors = plt.cm.Blues(np.linspace(0.3, 1, len(eps2_values)))
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel (a): s* vs gamma for different ε2
for i, eps2 in enumerate(eps2_values):
    axes[0].plot(
        gamma_grid,
        results_eps2[eps2],
        label=fr"$\varepsilon_2 = {eps2:.2f}$",
        color=colors[i],
        linewidth=3
    )

axes[0].set_xlabel(r"Risk aversion $\gamma$")
axes[0].set_ylabel(r"Optimal storage share $s^*$")
axes[0].set_title(r"(a) $s^*(\gamma)$ under different $\varepsilon_2$")
axes[0].legend(title=r"$\varepsilon_2$", loc="lower left")
axes[0].grid(True, linewidth=0.6)

# Panel (b): implied expected p2 for each ε2
eps_indices = np.arange(len(eps2_values))
p2_values = [expected_p2[eps] for eps in eps2_values]

bars = axes[1].bar(eps_indices, p2_values, color=colors, width=0.6)

axes[1].set_xticks(eps_indices)
axes[1].set_xticklabels(
    [fr"$\varepsilon_2={eps2:.2f}$" for eps2 in eps2_values]
)
axes[1].set_ylabel(r"Expected second-period price  $\mathbb{E}[p_2]$")
axes[1].set_title(
    r"(b) Implied $\mathbb{E}[p_2]$ under fixed "
    fr"$\mu={mu_fixed}$"
)
axes[1].grid(axis="y", linewidth=0.6)

# Add value labels above bars
for rect, val in zip(bars, p2_values):
    height = rect.get_height()
    axes[1].text(
        rect.get_x() + rect.get_width() / 2,
        height + 0.005,
        f"{val:.3f}",
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold'
    )

# Figure title
fig.suptitle(
    r"Impact of Second-Period Supply Elasticity on Storage and Expected Price"
    + "\n"
    + rf"($\mu={mu_fixed}$,  $Var(\theta_2)={variance}$,  $\theta_1={theta1}$,  $\kappa={kappa}$)",
    y=1.02
)

plt.tight_layout()
plt.savefig(
    os.path.join(target_dir, "sensitivity_to_gamma_eps2_with_p2_panel.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
