# Sensitivity panel rebuilt from scratch using an EXACT truncated-Poisson (support {1,...,9})
# No Monte Carlo, no clipping; we sum over the truncated support with proper renormalization.
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import os

# Global font settings
plt.rcParams.update({
    "axes.titlesize": 14,      # panel titles
    "axes.labelsize": 16,      # axis labels
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 13,     # legend entries
    "legend.title_fontsize": 14,
    "figure.titlesize": 20     # suptitle
})

# -----------------------------
# Parameters
# -----------------------------
kappa_fixed = 0.9
N1_fixed = 2                        # <-- N1 = 2, so p1 = 2/3
gamma_values = [0, 0.5, 2, 4, 7]
mu_params = np.arange(1, 9, 1)      # Poisson parameter; we'll compute E[N2] under truncation
s_grid = np.linspace(0, 1, 25)
support = np.arange(1, 10)          # support {1,...,9}

# -----------------------------
# Helpers: truncated Poisson and EU
# -----------------------------
def pois_pmf(n, mu):
    return np.exp(-mu) * (mu**n) / factorial(n)

def trunc_pois_pmf(mu, low=1, high=9):
    p = np.array([pois_pmf(n, mu) for n in support])
    mask = (support >= low) & (support <= high)
    Z = p[mask].sum()
    pmf = np.zeros_like(p)
    pmf[mask] = p[mask] / Z
    return pmf  # sums to 1 over {low,...,high}

def effective_price_levels():
    """
    Cournot price with N buyers, no outside option:
    p_t = N_t / (N_t + 1)
    """
    n = support.astype(float)
    return n / (n + 1.0)

def crra_utility(pi, gamma):
    with np.errstate(divide='ignore', invalid='ignore'):
        if gamma == 1:
            return np.log(pi)
        else:
            return (np.power(pi, 1 - gamma) - 1) / (1 - gamma)

def s_star_for(mu, gamma):
    pmf = trunc_pois_pmf(mu, 1, 9)
    prices = effective_price_levels()

    p1 = N1_fixed / (N1_fixed + 1.0)   # = 2/3

    # Risk neutral closed-form (binary corner) for gamma = 0
    if gamma == 0:
        Eprice = (pmf * prices).sum()
        return 1.0 if kappa_fixed * Eprice > p1 else 0.0

    # Otherwise compute expected utility on s-grid
    EU = []
    for s in s_grid:
        income1 = (1 - s) * p1
        income2 = s * kappa_fixed * prices
        income = income1 + income2
        EU.append((pmf * crra_utility(income, gamma)).sum())
    return s_grid[int(np.argmax(EU))]

def expected_trunc_mean(mu):
    pmf = trunc_pois_pmf(mu, 1, 9)
    return (pmf * support).sum()

# -----------------------------
# Compute s*(E[N2]) for each gamma
# -----------------------------
x_expected = np.array([expected_trunc_mean(mu) for mu in mu_params])  # x-axis values

results = {g: [] for g in gamma_values}
for mu in mu_params:
    for g in gamma_values:
        results[g].append(s_star_for(mu, g))

# -----------------------------
# Paths
# -----------------------------
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

# -----------------------------
# Plotting: single panel
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(gamma_values)))  # darker for higher gamma

for idx, g in enumerate(gamma_values):
    ax.plot(x_expected, results[g], label=f"$\\gamma$ = {g}",
            color=blues[idx], linewidth=2)

ax.axvline(x=N1_fixed, color='gray', linestyle='--', linewidth=1)
ax.text(N1_fixed + 0.05, 0.04, "$E[N_2] = N_1$", color='gray')

ax.set_xlabel("Expected $N_2$")
ax.set_ylabel("Optimal Storage Share $s^*$")
ax.set_title("Sensitivity of $s^*$ to Expected $N_2$ (Cournot, $\\kappa = 0.9$, $N_1 = 2$)")

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(title="Risk Aversion", loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(
    os.path.join(target_dir, "buyer_count_sensitivity_cournot.png"),
    dpi=300, bbox_inches="tight"
)
plt.show()
