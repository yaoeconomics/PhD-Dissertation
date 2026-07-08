"""
Optimal first-period storage share s*(kappa, gamma) when the second-period price
depends on a supply elasticity eps_2:  p_2(theta_2, eps_2) = eps_2 / (eps_2 + theta_2),
with theta_2 ~ Beta(mu, Var=0.05).

Restyled to match the visual-design language of `plot_storage_share_improved.py`:
same viridis + Normalize(0,1) colour system, 3D surfaces with a faint wireframe
overlay, upright bold kappa/gamma labels, a non-floating text2D s* label, a shared
horizontal colorbar, row block-labels for each elasticity, and a bold suptitle.

MODEL is unchanged (same constants, CRRA objective, corner solution at gamma = 0,
theta_2 drawn once per mu and reused across eps_2 to isolate the elasticity effect).
A fixed random seed is set so the rendered figure is reproducible; this does not
alter the model.

Output (into $OUTPUT_DIR, else ../../model_figures):
    3D_formulation_supply_elasticity.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import beta

# ---- styling (identical palette to the reference figure) ----
CMAP = cm.viridis
NORM = Normalize(0.0, 1.0)
ROW_COLORS = ["#31507d", "#4a7043", "#a5561f"]  # low -> high elasticity

plt.rcParams.update({"font.size": 15, "mathtext.fontset": "cm", "axes.titlesize": 15})

# --------------------------------
# Constants (unchanged)
# --------------------------------
theta1 = 0.5
p1 = 1 / (1 + theta1)
delta = 1.0

s_grid = np.linspace(0, 1, 100)
gamma_grid = np.linspace(0, 5, 50)
kappa_grid = np.linspace(0.6, 1.0, 50)
num_draws = 5000

means_modified = [0.2, 0.4, 0.5, 0.8]
var_theta2 = 0.05
eps2_values = [0.75, 1.00, 1.25]


# --------------------------------
# Helper functions (unchanged)
# --------------------------------
def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    return mu * factor, (1 - mu) * factor


def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)


def optimize_storage_share(theta2_draws, gamma, kappa, eps2):
    p2_draws = eps2 / (eps2 + theta2_draws)
    if gamma == 0:
        expected_p2 = np.mean(p2_draws)
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0
    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa * p2_draws
        utilities.append(np.mean(crra_utility(income, gamma)))
    return s_grid[np.argmax(utilities)]


# --------------------------------
# Simulation: surfaces[row (eps2)][col (mu)] = s*(gamma, kappa)
# --------------------------------
surfaces = [[None for _ in means_modified] for _ in eps2_values]
for col_idx, mu in enumerate(means_modified):
    alpha, beta_param = compute_beta_params(mu, var_theta2)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws, random_state=12345)
    for row_idx, eps2 in enumerate(eps2_values):
        surface = np.zeros((len(gamma_grid), len(kappa_grid)))
        for i, gamma in enumerate(gamma_grid):
            for j, kappa in enumerate(kappa_grid):
                surface[i, j] = optimize_storage_share(theta2_draws, gamma, kappa, eps2)
        surfaces[row_idx][col_idx] = surface

X, Y = np.meshgrid(kappa_grid, gamma_grid)

# --------------------------------
# Surface drawer (reference styling)
# --------------------------------
AX3D, TICK = 22, 15
ZOOM3D = 1.24


def draw_surface(ax, Z, mu, eps2):
    ax.plot_surface(X, Y, Z, cmap=CMAP, norm=NORM,
                    rcount=len(gamma_grid), ccount=len(kappa_grid),
                    linewidth=0, edgecolor="none", antialiased=True, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=12, cstride=12, color="k", linewidth=0.2, alpha=0.12)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel(r"$\kappa$", fontsize=AX3D, labelpad=8, fontweight="bold", rotation=0)
    ax.set_ylabel(r"$\gamma$", fontsize=AX3D, labelpad=10, fontweight="bold", rotation=0)
    ax.set_zlabel("")
    ax.text2D(0.05, 0.70, r"$s^{*}$", transform=ax.transAxes, fontsize=AX3D,
              fontweight="bold", ha="left", va="center")
    ax.set_title(fr"$\mu={mu},\ \varepsilon_2={eps2}$", fontsize=15, pad=2)
    ax.set_zlim(0, 1)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_yticks([0, 2.5, 5])
    ax.set_zticks([0, 0.5, 1])
    ax.tick_params(labelsize=TICK, pad=2)
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
# Figure: 3 rows (eps2) x 4 cols (mu)
# --------------------------------
fig = plt.figure(figsize=(22, 15))
gs = gridspec.GridSpec(len(eps2_values), len(means_modified),
                       left=0.065, right=0.985, top=0.905, bottom=0.135,
                       wspace=0.05, hspace=0.24)

row_cells = []
for row_idx, eps2 in enumerate(eps2_values):
    for col_idx, mu in enumerate(means_modified):
        ax = fig.add_subplot(gs[row_idx, col_idx], projection="3d")
        draw_surface(ax, surfaces[row_idx][col_idx], mu, eps2)
    row_cells.append(gs[row_idx, 0].get_position(fig))

# row block-labels (one per elasticity), on the left
for row_idx, eps2 in enumerate(eps2_values):
    cell = row_cells[row_idx]
    y_mid = (cell.y0 + cell.y1) / 2.0
    fig.text(0.014, y_mid, fr"$\varepsilon_2={eps2}$", rotation=90,
             va="center", ha="center", fontsize=20, fontweight="bold",
             color=ROW_COLORS[row_idx])

cax = fig.add_axes([0.32, 0.03, 0.38, 0.015])
sm = cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=16)
cb.set_label(r"optimal first-period storage share  $s^{*}$", fontsize=19, labelpad=8)

fig.suptitle(r"Optimal storage share $s^{*}(\kappa,\gamma)$ across mean buyer power $\mu$"
             r" and supply elasticity $\varepsilon_2$"
             "\n"
             r"$\theta_1=0.5$,  $\mathrm{Var}(\theta_2)=0.05$,  "
             r"$p_2=\varepsilon_2/(\varepsilon_2+\theta_2)$",
             fontsize=23, y=0.985, fontweight="bold")

# ---- save ----
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.environ.get("OUTPUT_DIR", os.path.join(grandparent_dir, "model_figures"))
os.makedirs(target_dir, exist_ok=True)
fig.savefig(os.path.join(target_dir, "3D_formulation_supply_elasticity_revised.png"),
            dpi=300, bbox_inches="tight")
plt.show()
