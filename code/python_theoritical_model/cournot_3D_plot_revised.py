"""
Optimal first-period storage share s*(kappa, gamma) under a Cournot second period
with a truncated-Poisson number of buyers N_2 in [1, 9].

Restyled to match the visual-design language of `plot_storage_share_improved.py`:
same viridis + Normalize(0,1) colour system, styled distribution panels with a
mean marker, 3D surfaces with a faint wireframe overlay, upright bold kappa/gamma
labels, a non-floating text2D s* label, a shared horizontal colorbar, and a bold
suptitle.

MODEL is unchanged (CRRA objective, Cournot price p = N/(N+1), N_1 = 2 so
p_1 = 2/3, closed-form risk-neutral benchmark at gamma = 0). A fixed random seed
is set so the rendered figure is reproducible; this does not alter the model.

Output (into $OUTPUT_DIR, else ../../model_figures):
    3D_cournot.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import poisson

np.random.seed(12345)  # reproducible truncated-Poisson draws (does not change the model)

# ---- styling (identical palette to the reference figure) ----
CMAP = cm.viridis
NORM = Normalize(0.0, 1.0)
BLUE = "#31507d"
ORANGE = "#a5561f"

plt.rcParams.update({"font.size": 15, "mathtext.fontset": "cm", "axes.titlesize": 15})


# --------------------------------
# Utility and pricing (unchanged)
# --------------------------------
def crra_utility(pi, gamma):
    with np.errstate(divide='ignore', invalid='ignore'):
        if gamma == 1:
            return np.log(pi)
        else:
            return (np.power(pi, 1 - gamma) - 1) / (1 - gamma)


def effective_price(N):
    """Cournot price with N buyers, no outside option: p_t = N_t / (N_t + 1)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return N / (N + 1)


# --------------------------------
# Optimal storage functions (unchanged)
# --------------------------------
def optimal_s(N2_samples, gamma, kappa):
    s_grid = np.linspace(0, 1, 25)
    expected_utils = []
    prices2 = effective_price(N2_samples)
    p1 = 2.0 / 3.0  # from N1 = 2
    for s in s_grid:
        income1 = (1 - s) * p1
        income2 = s * kappa * prices2
        income_total = income1 + income2
        EU = np.mean(crra_utility(income_total, gamma))
        expected_utils.append(EU)
    return s_grid[np.argmax(expected_utils)]


def closed_form_s(N2_samples, kappa):
    p1 = 2.0 / 3.0  # from N1 = 2
    income1 = p1
    prices2 = effective_price(N2_samples)
    income2 = kappa * np.mean(prices2)
    return 1.0 if income2 > income1 else 0.0


# --------------------------------
# Parameters (unchanged)
# --------------------------------
mu_values = [8, 6, 3]
gamma_grid = np.linspace(0, 5, 30)
kappa_grid = np.linspace(0.6, 1.0, 20)
s_min, s_max = 0.0, 1.0

# ---- precompute samples, PMFs and surfaces ----
samples, pmfs, surfaces = {}, {}, {}
x_vals = np.arange(1, 10)
for mu in mu_values:
    N2_samples = np.clip(np.random.poisson(mu, 5000), 1, 9)
    samples[mu] = N2_samples
    pmf_vals = poisson.pmf(x_vals, mu)
    pmf_vals = pmf_vals / pmf_vals.sum()  # renormalize over [1, 9]
    pmfs[mu] = pmf_vals

    s_star_surface = np.zeros((len(gamma_grid), len(kappa_grid)))
    for i, gamma in enumerate(gamma_grid):
        for j, kappa in enumerate(kappa_grid):
            if gamma == 0:
                s_star_surface[i, j] = closed_form_s(N2_samples, kappa)
            else:
                s_star_surface[i, j] = optimal_s(N2_samples, gamma, kappa)
    surfaces[mu] = s_star_surface

K, G = np.meshgrid(kappa_grid, gamma_grid)


# --------------------------------
# Panel drawers (reference styling)
# --------------------------------
def draw_pmf(ax, mu, show_ylabel):
    pmf_vals = pmfs[mu]
    mean_trunc = float(np.sum(x_vals * pmf_vals))
    ax.bar(x_vals, pmf_vals, width=0.62, color="#4C72B0", alpha=0.22,
           edgecolor=BLUE, linewidth=0.8, zorder=2)
    ax.plot(x_vals, pmf_vals, marker="o", ms=5, color=BLUE, lw=2.0, zorder=3)
    ax.axvline(mean_trunc, color="0.35", lw=1.2, ls=":")
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0, max(pmf_vals) * 1.18)
    ax.set_xticks(range(1, 10))
    ax.set_title(fr"$E(N_2)={mu}$   (Poisson, truncated $[1,9]$)", fontsize=18, pad=5)
    ax.set_xlabel(r"$N_2$  (second-period # of buyers)", fontsize=16, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("probability", fontsize=16, labelpad=4)
    ax.tick_params(labelsize=13)
    ax.grid(True, axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


AX3D, TICK = 24, 16
ZOOM3D = 1.28


def draw_surface(ax, Z, mu):
    ax.plot_surface(K, G, Z, cmap=CMAP, norm=NORM,
                    rcount=len(gamma_grid), ccount=len(kappa_grid),
                    linewidth=0, edgecolor="none", antialiased=True, alpha=1.0)
    ax.plot_wireframe(K, G, Z, rstride=6, cstride=6, color="k", linewidth=0.2, alpha=0.12)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel(r"$\kappa$", fontsize=AX3D, labelpad=9, fontweight="bold", rotation=0)
    ax.set_ylabel(r"$\gamma$", fontsize=AX3D, labelpad=11, fontweight="bold", rotation=0)
    ax.set_zlabel("")
    ax.text2D(0.05, 0.70, r"$s^{*}$", transform=ax.transAxes, fontsize=AX3D,
              fontweight="bold", ha="left", va="center")
    ax.set_title(fr"$s^{{*}}(\kappa,\gamma)\ \mid\ E(N_2)={mu}$", fontsize=18, pad=2)
    ax.set_zlim(s_min, s_max)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_yticks([0, 2.5, 5])
    ax.set_zticks([0, 0.5, 1])
    ax.tick_params(labelsize=TICK, pad=3)
    ax.view_init(elev=24, azim=-122)
    try:
        ax.set_box_aspect((1.1, 1.1, 0.78), zoom=ZOOM3D)
    except TypeError:
        ax.set_box_aspect((1.1, 1.1, 0.78))
    except Exception:
        pass
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.03)
        pane.set_edgecolor("0.85")
    ax.grid(True, alpha=0.25)


# --------------------------------
# Figure: 2 rows (PMF, surface) x 4 cols (mu)
# --------------------------------
fig = plt.figure(figsize=(17.5, 10.6))
gs = gridspec.GridSpec(2, 3, height_ratios=[0.85, 4.6],
                       hspace=0.30, wspace=0.06,
                       left=0.055, right=0.985, top=0.885, bottom=0.135)

for c, mu in enumerate(mu_values):
    draw_pmf(fig.add_subplot(gs[0, c]), mu, show_ylabel=(c == 0))
    draw_surface(fig.add_subplot(gs[1, c], projection="3d"), surfaces[mu], mu)

cax = fig.add_axes([0.32, 0.058, 0.38, 0.018])
sm = cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=16)
cb.set_label(r"optimal first-period storage share  $s^{*}$", fontsize=19, labelpad=8)

fig.suptitle(r"Optimal storage under Cournot competition"
             "\n"
             r"$s^{*}$ vs. risk aversion $\gamma$ and storage efficiency $\kappa$"
             r"  (truncated-Poisson buyers $N_2\in[1,9]$)",
             fontsize=24, y=0.985, fontweight="bold")

# ---- save ----
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.environ.get("OUTPUT_DIR", os.path.join(grandparent_dir, "model_figures"))
os.makedirs(target_dir, exist_ok=True)
fig.savefig(os.path.join(target_dir, "3D_cournot_revised.png"), dpi=300, bbox_inches="tight")
plt.show()
