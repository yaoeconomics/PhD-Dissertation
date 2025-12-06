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
# First-period price when ε1 is fixed at 1
p1_fixed = 1 / (1 + theta1)

kappa = 0.9
delta = 1.0
num_draws = 5000
variance = 0.05          # Var(theta_2)
s_grid = np.linspace(0, 1, 100)

# Set of second-period / common supply elasticities
eps2_values = [0.75, 1.00, 1.25]
eps_common_values = [0.75, 1.00, 1.25]

# Set of mu values for the grid figures (rows)
mu_values = [0.20, 0.30, 0.35]

# Risk-aversion grid
gamma_grid = np.linspace(0, 10, 100)

# --------------------------------
# Helper functions
# --------------------------------
def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

def optimize_s(theta2_draws, gamma, eps2, p1_value):
    """
    Case 1:
    θ2 ~ Beta(μ, variance)
    p1 is fixed (does not depend on ε2)
    p2(θ2, ε2) = ε2 / (ε2 + θ2)
    """
    p2_draws = eps2 / (eps2 + theta2_draws)

    if gamma == 0:
        expected_p2 = np.mean(p2_draws)
        return 1.0 if delta * kappa * expected_p2 > p1_value else 0.0

    utilities = []
    for s in s_grid:
        income = (1 - s) * p1_value + delta * s * kappa * p2_draws
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)
    return s_grid[np.argmax(utilities)]

def optimize_s_common(theta2_draws, gamma, eps_common):
    """
    Case 2: ε1 = ε2 = ε
    p1(ε) = ε / (ε + θ1)
    p2(θ2, ε) = ε / (ε + θ2)
    """
    p1_local = eps_common / (eps_common + theta1)
    p2_draws_local = eps_common / (eps_common + theta2_draws)

    # Risk-neutral benchmark
    if gamma == 0:
        expected_p2_local = np.mean(p2_draws_local)
        return 1.0 if delta * kappa * expected_p2_local > p1_local else 0.0

    utilities = []
    for s in s_grid:
        income = (1 - s) * p1_local + delta * s * kappa * p2_draws_local
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)

    return s_grid[np.argmax(utilities)]

# ============================================================
# Figure 1 grid: ε1 = 1, ε2 varies; rows = μ
# ============================================================

# Prepare color map for ε2
colors_eps2 = plt.cm.Blues(np.linspace(0.3, 1, len(eps2_values)))

fig1, axes1 = plt.subplots(len(mu_values), 2, figsize=(18, 6 * len(mu_values)))

# Ensure axes1 is 2D array even if len(mu_values)=1
if len(mu_values) == 1:
    axes1 = np.array([axes1])

for row_idx, mu in enumerate(mu_values):
    # Draw θ2 for this μ
    alpha, beta_param = compute_beta_params(mu, variance)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)

    # Storage results and expected prices for this μ
    results_eps2 = {eps2: [] for eps2 in eps2_values}
    expected_p2 = {}

    # Compute s*(γ) and E[p2] for each ε2
    for eps2 in eps2_values:
        p2_draws = eps2 / (eps2 + theta2_draws)
        expected_p2[eps2] = np.mean(p2_draws)

        for gamma in gamma_grid:
            s_star = optimize_s(theta2_draws, gamma, eps2, p1_fixed)
            results_eps2[eps2].append(s_star)

    # --- Panel (a) in this row: s*(γ) for given μ ---
    ax_left = axes1[row_idx, 0]
    for i, eps2 in enumerate(eps2_values):
        ax_left.plot(
            gamma_grid,
            results_eps2[eps2],
            label=fr"$\varepsilon_2 = {eps2:.2f}$",
            color=colors_eps2[i],
            linewidth=3
        )

    ax_left.set_xlabel(r"Risk aversion $\gamma$")
    if row_idx == 0:
        ax_left.set_title(r"$s^*(\gamma)$ under different $\varepsilon_2$")
    ax_left.set_ylabel(r"$s^*$" + f"\n($E[\\theta_2]={mu:.2f}$)")
    ax_left.grid(True, linewidth=0.6)
    ax_left.set_ylim(0, 1.02)
    if row_idx == 0:
        ax_left.legend(loc="upper right")

    # --- Panel (b) in this row: bar chart of E[p2] ---
    ax_right = axes1[row_idx, 1]
    eps_indices = np.arange(len(eps2_values))
    p2_values = [expected_p2[eps] for eps in eps2_values]

    bars = ax_right.bar(eps_indices, p2_values, color=colors_eps2, width=0.6)

    ax_right.set_xticks(eps_indices)
    ax_right.set_xticklabels(
        [fr"$\varepsilon_2={eps2:.2f}$" for eps2 in eps2_values]
    )
    if row_idx == 0:
        ax_right.set_title(
            r"$\mathbb{E}[p_2]$ under different $\varepsilon_2$"
        )
    ax_right.set_ylabel(
        r"$\mathbb{E}[p_2]$" + f"\n($E[\\theta_2]={mu:.2f}$)"
    )
    ax_right.grid(axis="y", linewidth=0.6)

    # Add value labels above bars
    for rect, val in zip(bars, p2_values):
        height = rect.get_height()
        ax_right.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.005,
            f"{val:.3f}",
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold'
        )

fig1.suptitle(
    r"Impact of Second-Period Supply Elasticity on Storage and Expected Price"
    + "\n"
    + rf"($E[\theta_2]\in\{{0.20,0.35,0.45\}},\ Var(\theta_2)={variance},\ \theta_1={theta1},\ \kappa={kappa},\ \varepsilon_1=1$)",
    y=1.02
)

plt.tight_layout()
plt.savefig(
    os.path.join(target_dir, "grid_sensitivity_gamma_eps2_by_mu.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Figure 2 grid: ε1 = ε2 = ε; rows = μ
# ============================================================

colors_common = plt.cm.Greens(np.linspace(0.3, 1, len(eps_common_values)))
fig2, axes2 = plt.subplots(len(mu_values), 2, figsize=(18, 6 * len(mu_values)))

# Ensure axes2 is 2D
if len(mu_values) == 1:
    axes2 = np.array([axes2])

for row_idx, mu in enumerate(mu_values):
    # Draw θ2 for this μ
    alpha, beta_param = compute_beta_params(mu, variance)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)

    results_eps_common = {eps: [] for eps in eps_common_values}
    expected_p2_common = {}
    p1_values = {}

    # Compute s*(γ), p1(ε), and E[p2(ε)] for each common ε
    for eps in eps_common_values:
        p1_local = eps / (eps + theta1)
        p2_draws_local = eps / (eps + theta2_draws)

        p1_values[eps] = p1_local
        expected_p2_common[eps] = np.mean(p2_draws_local)

        for gamma in gamma_grid:
            s_star = optimize_s_common(theta2_draws, gamma, eps)
            results_eps_common[eps].append(s_star)

    # --- Panel (a) in this row: s*(γ) with ε1 = ε2 = ε ---
    ax_left = axes2[row_idx, 0]
    for i, eps in enumerate(eps_common_values):
        ax_left.plot(
            gamma_grid,
            results_eps_common[eps],
            label=fr"$\varepsilon_1 = \varepsilon_2 = {eps:.2f}$",
            color=colors_common[i],
            linewidth=3
        )

    ax_left.set_xlabel(r"Risk aversion $\gamma$")
    if row_idx == 0:
        ax_left.set_title(
            r"$s^*(\gamma)$ when $p_1$ and $p_2$ both depend on $\varepsilon$"
        )
    ax_left.set_ylabel(r"$s^*$" + f"\n($E[\\theta_2]={mu:.2f}$)")
    ax_left.grid(True, linewidth=0.6)
    ax_left.set_ylim(0, 1.02)
    if row_idx == 0:
        ax_left.legend(loc="upper right")

    # --- Panel (b) in this row: bar plot of p1(ε) and E[p2(ε)] ---
    ax_right = axes2[row_idx, 1]
    eps_indices = np.arange(len(eps_common_values))
    width = 0.35

    p1_list = [p1_values[eps] for eps in eps_common_values]
    p2_list = [expected_p2_common[eps] for eps in eps_common_values]

    bars1 = ax_right.bar(
        eps_indices - width / 2,
        p1_list,
        width=width,
        label=r"$p_1(\varepsilon)$",
        color=colors_common,
        alpha=0.8
    )

    bars2 = ax_right.bar(
        eps_indices + width / 2,
        p2_list,
        width=width,
        label=r"$\mathbb{E}[p_2(\varepsilon)]$",
        color=colors_common,
        alpha=0.4
    )

    ax_right.set_xticks(eps_indices)
    ax_right.set_xticklabels(
        [fr"$\varepsilon = {eps:.2f}$" for eps in eps_common_values]
    )
    if row_idx == 0:
        ax_right.set_title(
            r"$p_1(\varepsilon)$ and $\mathbb{E}[p_2(\varepsilon)]$"
        )
    ax_right.set_ylabel("Price" + f"\n($E[\\theta_2]={mu:.2f}$)")
    ax_right.grid(axis="y", linewidth=0.6)
    if row_idx == 0:
        ax_right.legend(loc="lower right")

    # Add value labels
    for rect in list(bars1) + list(bars2):
        height = rect.get_height()
        ax_right.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

fig2.suptitle(
    r"Common Supply Elasticity Across Periods: "
    r"$p_t = \frac{\varepsilon}{\varepsilon + \theta_t}$"
    + "\n"
    + rf"($E[\theta_2]\in\{{0.20,0.35,0.45\}},\ Var(\theta_2)={variance},\ \theta_1={theta1},\ \kappa={kappa}$)",
    y=1.02
)

plt.tight_layout()
plt.savefig(
    os.path.join(
        target_dir,
        "grid_sensitivity_gamma_eps_common_by_mu.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
