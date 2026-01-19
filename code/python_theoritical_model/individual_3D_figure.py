import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import os

# ----------------------------
# Global font / style settings
# ----------------------------
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,      # axis labels larger
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22
})

# ----------------------------
# Model constants / grids
# ----------------------------
theta1 = 0.5
p1 = 1 / (1 + theta1)
delta = 1.0

s_grid = np.linspace(0, 1, 100)      # storage share choices
gamma_grid = np.linspace(0, 10, 50)  # risk aversion
kappa_grid = np.linspace(0.6, 1.0, 50)  # storage efficiency
num_draws = 5000

# Target case
mu = 0.2
sigma2 = 0.05

# ----------------------------
# Helper functions
# ----------------------------
def compute_beta_params(mu_, sigma2_):
    # Validity: need sigma2 < mu(1-mu)
    max_var = mu_ * (1 - mu_)
    if sigma2_ <= 0 or sigma2_ >= max_var:
        raise ValueError(
            f"Invalid variance for Beta: need 0 < sigma2 < mu(1-mu)={max_var:.4f}, got {sigma2_}."
        )
    factor = mu_ * (1 - mu_) / sigma2_ - 1
    alpha_ = mu_ * factor
    beta_ = (1 - mu_) * factor
    return alpha_, beta_

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(pi)
    return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_storage_share(theta2_draws, gamma, kappa):
    # Risk-neutral shortcut
    if np.isclose(gamma, 0.0):
        expected_p2 = np.mean(1 / (1 + theta2_draws))
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0

    utilities = np.empty_like(s_grid)
    for idx, s in enumerate(s_grid):
        income = (1 - s) * p1 + delta * s * kappa / (1 + theta2_draws)
        utilities[idx] = np.mean(crra_utility(income, gamma))
    return s_grid[np.argmax(utilities)]

# ----------------------------
# Draw theta2 and compute s*
# ----------------------------
alpha, beta_param = compute_beta_params(mu, sigma2)
theta2_draws = beta.rvs(alpha, beta_param, size=num_draws, random_state=123)

surface = np.zeros((len(gamma_grid), len(kappa_grid)))
for i, gamma in enumerate(gamma_grid):
    for j, kappa in enumerate(kappa_grid):
        surface[i, j] = optimize_storage_share(theta2_draws, gamma, kappa)

# ----------------------------
# Plot single 3D surface
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

X, Y = np.meshgrid(kappa_grid, gamma_grid)  # X=kappa, Y=gamma
Z = surface

ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_title(rf"Optimal storage share $s^*$  ($\theta_1$=0.5, E[$\theta_2$]={mu}, Var[$\theta_2$]={sigma2})")
ax.set_xlabel(r"$\kappa$ (storage efficiency)", labelpad=12)
ax.set_ylabel(r"$\gamma$ (risk aversion)", labelpad=12)
ax.set_zlabel(r"$s^*$", labelpad=12)
ax.set_zlim(0, 1)

# Make tick labels bigger on the 3D axis too
ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='z', which='major', labelsize=16)

# View angle (adjust if you like)
ax.view_init(elev=30, azim=-135)

plt.tight_layout()

# ----------------------------
# Save to .../model_figures/3D_formulation_mu0p2_var0p05.png
# (works in scripts; also works in notebooks by falling back to cwd)
# ----------------------------
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

out_path = os.path.join(target_dir, "individual_3D_formulation_mu0p2_var0p05.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
