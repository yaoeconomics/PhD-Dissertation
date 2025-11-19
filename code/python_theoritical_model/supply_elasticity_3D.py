import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Global font settings
plt.rcParams.update({
    "axes.titlesize": 14,      # panel titles
    "axes.labelsize": 14,      # axis labels
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 14,     # legend entries
    "legend.title_fontsize": 14,
    "figure.titlesize": 20     # suptitle
})

# --------------------------------
# Paths
# --------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

# --------------------------------
# Constants
# --------------------------------
theta1 = 0.5
# First-period price: normalized ε1 = 1
p1 = 1 / (1 + theta1)
delta = 1.0

s_grid = np.linspace(0, 1, 100)
gamma_grid = np.linspace(0, 10, 50)
kappa_grid = np.linspace(0.6, 1.0, 50)
num_draws = 5000

# Means for θ2
means_modified = [0.2, 0.4, 0.5, 0.8]
# Fixed variance of θ2
var_theta2 = 0.05
# Second-period supply elasticities: one per row
eps2_values = [0.75, 1.00, 1.25]

# --------------------------------
# Helper functions
# --------------------------------
def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_storage_share(theta2_draws, gamma, kappa, eps2):
    """
    θ2 ~ Beta(μ, var_theta2)
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

# --------------------------------
# Simulation: build 3D surfaces
# surfaces[row_idx][col_idx] = s*(γ, κ) for given (ε2, μ)
# --------------------------------
surfaces = [[None for _ in means_modified] for _ in eps2_values]

for col_idx, mu in enumerate(means_modified):
    # Draw θ2 once for each μ, reused across ε2 to isolate elasticity effect
    alpha, beta_param = compute_beta_params(mu, var_theta2)
    theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)

    for row_idx, eps2 in enumerate(eps2_values):
        surface = np.zeros((len(gamma_grid), len(kappa_grid)))
        for i, gamma in enumerate(gamma_grid):
            for j, kappa in enumerate(kappa_grid):
                s_star = optimize_storage_share(theta2_draws, gamma, kappa, eps2)
                surface[i, j] = s_star
        surfaces[row_idx][col_idx] = surface

# --------------------------------
# Plotting: 3 rows (ε2), 4 cols (μ)
# --------------------------------
fig = plt.figure(figsize=(22, 13))  # big enough canvas

gs = fig.add_gridspec(
    nrows=len(eps2_values),
    ncols=len(means_modified),
    left=0.04,
    right=0.98,
    top=0.90,
    bottom=0.06,
    wspace=0.15,    # expanded for readable axis labels
    hspace=0.40
)

X, Y = np.meshgrid(kappa_grid, gamma_grid)

for row_idx, eps2 in enumerate(eps2_values):
    for col_idx, mu in enumerate(means_modified):
        ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
        Z = surfaces[row_idx][col_idx]

        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
        ax.set_zlim(0, 1)
        ax.view_init(elev=30, azim=-135)

        # Panel title
        ax.set_title(
            rf"$\mu={mu}$, $\varepsilon_2={eps2}$",
            pad=8
        )

        # --- Make axis labels appear in ALL panels ---
        ax.set_xlabel(r"$\kappa$", labelpad=6)
        ax.set_ylabel(r"$\gamma$", labelpad=6)
        ax.set_zlabel(r"$s^*$", labelpad=6)

fig.suptitle(
    r"Optimal Storage Share $s^*$ over $(\gamma,\kappa)$ "
    r"for Different $\mu$ and $\varepsilon_2$"
    "\n"
    r"$\theta_1=0.5$, $\mathrm{Var}(\theta_2)=0.05$",
    fontsize=20,
    y=0.97
)

fig.savefig(
    os.path.join(target_dir, "3D_formulation_supply_elasticity.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
