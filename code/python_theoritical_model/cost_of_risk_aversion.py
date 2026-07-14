import os
from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.lines import Line2D

# ---------------------- Output directory ----------------------
try:
    SAVE_DIR = Path(__file__).resolve().parents[2] / "model_figures"
except NameError:  # e.g., notebook
    SAVE_DIR = Path.cwd() / "model_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
target_dir = str(SAVE_DIR)

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16
})

# Single, consistent axis label for the expected second-period buyer-power parameter.
MU_LABEL = r"Expected second-period buyer power, $\mu_{\theta_2}$"

# ---------------------- Primitives & Grids ----------------------
theta1, delta = 0.5, 1.0
p1 = 1 / (1 + theta1)
variance = 0.05

mu_values = np.arange(0.10, 0.601, 0.01)   # grid over mu_{theta_2} = E[theta_2]
gammas    = [1, 2, 4]                       # risk-aversion levels
kappas    = [0.80, 0.90, 0.95]             # storage efficiency
num_draws = 20000
s_grid    = np.linspace(0, 1, 50)

kappa_baseline = 0.90                       # used by Figure 3

rng = np.random.default_rng(20260709)       # fixed seed: identical draws across kappa

# ---------------------- Helpers (kappa is an explicit argument) ----------------------
def beta_params(mu, var):
    f = mu * (1 - mu) / var - 1
    return mu * f, (1 - mu) * f

def U_crra(pi, gamma):
    pi = np.maximum(pi, 1e-10)
    return np.log(pi) if gamma == 1 else (pi**(1 - gamma) - 1) / (1 - gamma)

def s_star_RN(v_mean, kappa):
    return 1.0 if delta * kappa * v_mean > p1 else 0.0

def s_star_RA(v_draws, gamma, kappa):
    incomes = (1 - s_grid)[:, None] * p1 + (delta * s_grid[:, None] * kappa) * v_draws[None, :]
    util = U_crra(incomes, gamma).mean(axis=1)
    return s_grid[util.argmax()]

def E_income_from_v(s, v_mean, kappa):
    return (1 - s) * p1 + delta * s * kappa * v_mean

# ---------------------- Simulation ----------------------
G, M, K = len(gammas), len(mu_values), len(kappas)
s_RA    = np.zeros((K, G, M))
s_RN    = np.zeros((K, M))
gap_abs = np.zeros((K, G, M))
gap_pct = np.zeros((K, G, M))

for j, mu in enumerate(mu_values):
    a, b = beta_params(mu, variance)
    # Draw once per mu so every kappa series faces the same realized price risk
    th2  = beta.rvs(a, b, size=num_draws, random_state=rng)
    v    = 1.0 / (1.0 + th2)
    vbar = v.mean()

    for k, kappa in enumerate(kappas):
        s_rn = s_star_RN(vbar, kappa)
        s_RN[k, j] = s_rn
        Ei_rn = E_income_from_v(s_rn, vbar, kappa)

        for i, g in enumerate(gammas):
            s_ra = s_star_RA(v, g, kappa)
            s_RA[k, i, j] = s_ra
            Ei_ra = E_income_from_v(s_ra, vbar, kappa)
            diff  = Ei_rn - Ei_ra
            gap_abs[k, i, j] = diff
            gap_pct[k, i, j] = 0.0 if Ei_rn <= 1e-12 else 100.0 * diff / Ei_rn

# ---------------------- Visual encoding ----------------------
# Color     -> gamma (risk aversion): high-contrast, colorblind-safe hues (Okabe-Ito).
# Linestyle -> kappa (storage efficiency): solid / dashed / dotted are maximally distinct,
#              so kappa=0.90 and kappa=0.95 no longer read as near-identical dashes.
# Categorical set: gamma=1 yellow (gold, so it stays legible on white),
# gamma=2 green, gamma=4 black.
# gamma colors are set PER FIGURE.
# Figure 1: yellow (gold, legible on white), green, black.
GAMMA_COLORS_FIG1 = ["#E6A700", "#2CA02C", "#000000", "#7f7f7f", "#c7c7c7"][:G]
# Figure 2: dark -> light blue ramp (gamma=1 darkest, gamma=4 lightest).
GAMMA_COLORS_BLUE = ["#08306b", "#2b6cb0", "#7fb3e0", "#a8cff0", "#cfe4f7"][:G]

STYLE_POOL = ["-", (0, (6, 2)), (0, (1, 1.6)), (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1))]
WIDTH_POOL = [2.6, 2.2, 2.0, 1.6, 1.3]
kappa_styles = {kap: st for kap, st in zip(kappas, cycle(STYLE_POOL))}
kappa_widths = {kap: w for kap, w in zip(kappas, cycle(WIDTH_POOL))}

def gamma_handles(colors, extra=None):
    h = [Line2D([0], [0], color=colors[i], lw=2.6, ls="-", label=rf"$\gamma={g}$")
         for i, g in enumerate(gammas)]
    if extra is not None:
        h.append(extra)
    return h

def kappa_handles():
    return [Line2D([0], [0], color="0.25", lw=kappa_widths[kap], ls=kappa_styles[kap],
                   label=rf"$\kappa={kap:.2f}$") for kap in kappas]

# ---------------------- Figure 1: Relative (%) income gap ----------------------
fig, ax = plt.subplots(figsize=(11, 6))

for k, kappa in enumerate(kappas):
    for i, g in enumerate(gammas):
        ax.plot(mu_values, gap_pct[k, i],
                color=GAMMA_COLORS_FIG1[i],
                ls=kappa_styles[kappa],
                lw=kappa_widths[kappa])

ax.set_xlabel(MU_LABEL)
ax.set_ylabel(r"Percentage relative income gap")
ax.grid(True, alpha=0.3)

leg1 = ax.legend(handles=gamma_handles(GAMMA_COLORS_FIG1), title="Risk aversion",
                 loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.add_artist(leg1)
ax.legend(handles=kappa_handles(), title="Storage efficiency",
          loc="upper left", bbox_to_anchor=(1.02, 0.55), frameon=False)

fig.tight_layout()
plt.savefig(os.path.join(target_dir, "income_gap_vs_mu_(unified)_revised.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# ---------------------- Figure 2: Sensitivity of s* to mu_{theta_2} ----------------------
fig, ax = plt.subplots(figsize=(11, 6))

for k, kappa in enumerate(kappas):
    ax.plot(mu_values, s_RN[k], color="k", ls=kappa_styles[kappa], lw=1.4, alpha=0.6)
    for i, g in enumerate(gammas):
        ax.plot(mu_values, s_RA[k, i],
                color=GAMMA_COLORS_BLUE[i],
                ls=kappa_styles[kappa],
                lw=kappa_widths[kappa])

ax.set_xlabel(MU_LABEL)
ax.set_ylabel(r"Optimal storage share $s^*$")
ax.grid(True, alpha=0.3)

rn_handle = Line2D([0], [0], color="k", lw=1.4, alpha=0.6, label=r"Risk-neutral $s^*$")
leg1 = ax.legend(handles=gamma_handles(GAMMA_COLORS_BLUE, extra=rn_handle), title="Risk aversion",
                 loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.add_artist(leg1)
ax.legend(handles=kappa_handles(), title="Storage efficiency",
          loc="upper left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
plt.savefig(os.path.join(target_dir, "sensitivity_to_theta_2.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# ---------------------- Figure 3: Expected Price Gain vs mu_{theta_2} ----------------------
E_p2 = np.zeros(M)
for j, mu in enumerate(mu_values):
    a, b = beta_params(mu, variance)
    th2  = beta.rvs(a, b, size=num_draws, random_state=rng)
    E_p2[j] = (1.0 / (1.0 + th2)).mean()

theta1_vals = [0.25, 0.50, 0.75]
p1_vals = [1.0 / (1.0 + t1) for t1 in theta1_vals]
gains = [kappa_baseline * E_p2 - p1v for p1v in p1_vals]

t1_colors = {0.25: "#bdd7e7", 0.50: "#6baed6", 0.75: "#08519c"}
t1_styles = {0.25: "--", 0.50: "-", 0.75: "-."}

plt.figure(figsize=(10, 6))

for t1, g in zip(theta1_vals, gains):
    plt.plot(mu_values, g,
             color=t1_colors[t1], ls=t1_styles[t1],
             lw=3 if t1 == 0.50 else 2,
             label=rf"$\theta_1 = {t1:.2f}$  ($p_1={1/(1+t1):.3f}$)")

    signs = np.sign(g)
    idxs = np.where(signs[:-1] * signs[1:] <= 0)[0]
    for idx in idxs:
        x0, x1 = mu_values[idx], mu_values[idx + 1]
        y0, y1 = g[idx], g[idx + 1]
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        plt.scatter(root, 0.0, color=t1_colors[t1], edgecolor="black", s=55, zorder=5)
        plt.text(root, -0.012, f"{root:.3f}", color=t1_colors[t1],
                 fontsize=16, fontweight="bold", ha="center", va="top")

plt.axhline(0.0, lw=2.5, color="darkred", alpha=0.7)
plt.text(mu_values[-1] * 0.98, 0.005,
         r"Indifference: $\kappa\,\mathbb{E}[p_2]=p_1$",
         fontsize=14, color="red", ha="right", va="bottom")

plt.xlabel(MU_LABEL)
plt.ylabel(r"Expected Price Gain  $=\kappa\,\mathbb{E}[p_2]-p_1$")
plt.grid(True, alpha=0.3)
plt.legend(title="First-period buyer power", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "expected_price_gain_vs_mu.png"),
            dpi=300, bbox_inches="tight")
plt.show()

print("target_dir:", target_dir)
print("files:", sorted(os.listdir(target_dir)))