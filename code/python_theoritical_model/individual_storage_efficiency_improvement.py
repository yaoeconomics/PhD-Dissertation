import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
import pandas as pd
import itertools
import string



import os

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上一级目录
parent_dir = os.path.dirname(current_dir)
# 上两级目录
grandparent_dir = os.path.dirname(parent_dir)
# 上两级平行目录（例如 results）
target_dir = os.path.join(grandparent_dir, "model_figures")

os.makedirs(target_dir, exist_ok=True)

# -------------------------------
# Settings
# -------------------------------
policy_increase = 1.10
delta = 1.0
variance = 0.02
thetas_mu = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
kappas0 = [0.70, 0.80, 0.90]
gammas = [0, 0.5, 2, 4]
num_draws = 100000
s_grid = np.linspace(0, 1, 41)
rng = np.random.default_rng(20251014)

scenarios = [
    {"name": "Symmetric Buyer Power (θ₁ = E[θ₂])", "short": "Symmetric", "theta1_fn": lambda mu: mu},
    {"name": "Fixed Current Buyer Power (θ₁ = 0.5)", "short": "Fixed θ₁=0.5", "theta1_fn": lambda mu: 0.5},
]

# -------------------------------
# Helper functions
# -------------------------------
def feasible_beta_params(mu, sigma2, eps=1e-9):
    max_var = mu * (1 - mu)
    if sigma2 >= max_var:
        sigma2 = max_var * 0.999 - eps
    factor = mu * (1 - mu) / sigma2 - 1.0
    alpha = mu * factor
    beta_p = (1 - mu) * factor
    return max(alpha, eps), max(beta_p, eps)

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if gamma == 1:
        return np.log(pi)
    return (pi ** (1 - gamma) - 1) / (1 - gamma)

def p_from_theta(theta):
    return 1.0 / (1.0 + theta)

def optimize_s(theta2_draws, p1, kappa, gamma, s_grid):
    if gamma == 0:
        exp_p2 = np.mean(1.0 / (1.0 + theta2_draws))
        return float(1.0) if delta * kappa * exp_p2 > p1 else float(0.0)
    v_draws = 1.0 / (1.0 + theta2_draws)
    incomes = (1 - s_grid[:, None]) * p1 + s_grid[:, None] * kappa * v_draws
    utils = crra_utility(incomes, gamma).mean(axis=1)
    tol = 1e-10
    best_u = utils.max()
    idx = int(np.where(np.isclose(utils, best_u, atol=tol))[0].min())
    return float(s_grid[idx])

# -------------------------------
# Simulation
# -------------------------------
records = []
for mu in thetas_mu:
    a, b = feasible_beta_params(mu, variance)
    theta2_draws = beta_dist(a, b).rvs(size=num_draws, random_state=rng)
    v_draws = 1.0 / (1.0 + theta2_draws)

    for k0 in kappas0:
        k1 = min(1.0, policy_increase * k0)

        for scenario in scenarios:
            theta1 = scenario["theta1_fn"](mu)
            p1 = p_from_theta(theta1)

            for gamma in gammas:
                s0 = optimize_s(theta2_draws, p1, k0, gamma, s_grid)
                income0 = (1 - s0) * p1 + s0 * k0 * v_draws
                Epi0 = income0.mean()
                EU0 = Epi0 if gamma == 0 else crra_utility(income0, gamma).mean()

                s1 = optimize_s(theta2_draws, p1, k1, gamma, s_grid)
                income1 = (1 - s1) * p1 + s1 * k1 * v_draws
                Epi1 = income1.mean()
                EU1 = Epi1 if gamma == 0 else crra_utility(income1, gamma).mean()

                records.append({
                    "scenario": scenario["name"],
                    "scenario_short": scenario["short"],
                    "mu=E[theta2]": mu,
                    "theta1": theta1,
                    "p1": p1,
                    "kappa0": k0,
                    "kappa1": k1,
                    "gamma": gamma,
                    "s*_before": s0,
                    "s*_after": s1,
                    "Δs*": s1 - s0,
                    "E[π]_before": Epi0,
                    "E[π]_after": Epi1,
                    "ΔE[π]": Epi1 - Epi0,
                    "EU_before": EU0,
                    "EU_after": EU1,
                    "ΔEU": EU1 - EU0
                })

df = pd.DataFrame.from_records(records)
df.to_csv(os.path.join(target_dir, "individual_storage_efficiency_policy_results.csv"), index=False)

# -------------------------------
# Styling
# -------------------------------
import matplotlib.cm as cm
cmap = cm.get_cmap("Blues")
def deeper_blue(val):
    frac = 0.50 + 0.60 * (val - min(gammas)) / (max(gammas) - min(gammas))
    return cmap(frac)
color_map = {g: deeper_blue(g) for g in gammas}
linestyles = {0: "-", 0.5: "--", 2: "-.", 4: ":"}
markers = {0: "o", 0.5: "s", 2: "D", 4: "^"}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -------------------------------
# NEW GRID: ONLY LAST TWO ROWS
#   Rows (2): [ΔE[π] | Fixed], [Δs* | Fixed]
#   Cols (3): κ0 in {0.70, 0.80, 0.90}
# -------------------------------
row_defs = [
    ("ΔE[π]", "Fixed Current Buyer Power (θ₁ = 0.5)"),
    ("Δs*",  "Fixed Current Buyer Power (θ₁ = 0.5)"),
]

n_rows, n_cols = 2, len(kappas0)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8*n_cols, 2.6*n_rows),
                         sharex=False, sharey=False)

# y-limits per metric
limits = {}
for metric in ["ΔE[π]", "Δs*"]:
    limits[metric] = (float(df[metric].min()), float(df[metric].max()))

def padded_limits(lo, hi, pct=0.06):
    span = hi - lo
    if np.isclose(span, 0):
        span = 1e-6
    pad = pct * span
    return lo - pad, hi + pad

panel_labels = list(string.ascii_lowercase)

# -------------------------------
# Plotting
# -------------------------------
for r, (metric, scenario_name) in enumerate(row_defs):
    scenario_df = df[df["scenario"] == scenario_name]
    ylo, yhi = padded_limits(*limits[metric])

    for c, k0 in enumerate(kappas0):
        ax = axes[r, c]
        sub_k = scenario_df[scenario_df["kappa0"] == k0].copy()
        k1 = min(1.0, policy_increase * k0)

        for gamma in gammas:
            sub_g = sub_k[sub_k["gamma"] == gamma].sort_values("mu=E[theta2]")
            x = sub_g["mu=E[theta2]"].values
            y = sub_g[metric].values
            ax.plot(
                x, y,
                marker=markers[gamma],
                linewidth=2,
                markersize=4,
                linestyle=linestyles[gamma],
                color=color_map[gamma],
                label=f"γ={gamma}"
            )

        if r == 0:
            ax.set_title(f"κ: {k0:.2f} → {k1:.2f}")

        if c == 0:
            left_label = "E[π]" if metric == "ΔE[π]" else "Δ s*"
            ax.set_ylabel(left_label + "\n(Fixed θ₁=0.5)")

        ax.set_xlabel("μ = E[θ₂]")
        ax.grid(True, alpha=0.35)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        ax.set_ylim(ylo, yhi)

        label_idx = r*n_cols + c
        ax.text(0.01, 0.98, f"({panel_labels[label_idx]})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")

# Unified legend
handles, labels = axes[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(gammas), loc="lower center",
           frameon=False, bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(target_dir,
           "individual_storage_efficiency_policy_grid.png"),
           dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved figure: individual_storage_efficiency_policy_grid.png")
