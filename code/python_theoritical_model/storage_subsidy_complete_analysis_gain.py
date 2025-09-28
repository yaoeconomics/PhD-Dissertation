import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss
from PIL import Image

# -----------------------------
# Paths (robust to interactive)
# -----------------------------
try:
    CURRENT_FILE = os.path.abspath(__file__)
    current_dir = os.path.dirname(CURRENT_FILE)
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

# -----------------------------
# Global settings
# -----------------------------
REUSE_WORLDS = True
STORE_THRESHOLD = 0.01   # 1% counts as storing

rng = np.random.default_rng(314)

# Grids and parameters
mu1_grid = np.arange(0.10, 0.81, 0.1)  # Mean of the Distribution of θ1 (μ1)
sigma2_1 = 0.1                       # Variance for θ1 Beta
sigma2_2 = 0.1                      # Variance for θ2 Beta
N = 100
# gammas = 10.0 * rng.beta(1, 5, size=N); gammas.sort()
gammas = rng.uniform(0, 10, size=N); gammas.sort()

kappa_scenarios = [0.80, 0.85, 0.90, 0.95]
kappa_pairs = [(0.80, 0.85), (0.80, 0.90), (0.80, 0.95)]
R = 2000
nodes, weights = leggauss(12)
x_nodes = 0.5*(nodes+1); w_nodes = 0.5*weights

theta1_grid = np.linspace(0.001, 0.999, 30)
gamma_grid  = np.linspace(0.0, 10.0, 30)

# Column scenarios: μ2 = μ1 - gap
scenarios = [
    ("Baseline: μ₂ = μ₁", 0.00),
    ("μ₂ = μ₁ − 0.10", 0.10),
    ("μ₂ = μ₁ − 0.20", 0.20),
    ("μ₂ = μ₁ − 0.40", 0.40),
]

# -----------------------------
# Utility and helpers
# -----------------------------
def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(pi)
    return (pi**(1-gamma) - 1) / (1-gamma)

def crra_u_inv(u, gamma):
    if abs(gamma - 1.0) < 1e-12:
        return np.exp(u)
    return np.maximum(u*(1-gamma)+1, 1e-12)**(1/(1-gamma))

def alpha_beta_from(mu, sigma2):
    # Ensure feasible sigma2
    eps = 1e-9
    mu = np.clip(mu, eps, 1-eps)
    max_var = mu*(1-mu)
    if sigma2 >= max_var:
        sigma2 = 0.95 * max_var
    factor = mu*(1-mu)/sigma2 - 1
    return mu*factor, (1-mu)*factor

def golden_max(f, a=0.0, b=1.0, tol=5e-5, max_iter=90):
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
    return 0.5*(a + b)

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes):
    p1 = 1.0/(1.0 + theta1)
    if abs(gamma) < 1e-12:
        E_inv = float(np.sum(1.0/(1.0 + theta2_nodes) * w_nodes))
        return 1.0 if (kappa*E_inv > p1) else 0.0
    def EU_of_s(s):
        inc = (1 - s)*p1 + s*kappa/(1.0 + theta2_nodes)
        if abs(gamma - 1.0) < 1e-12:
            u = np.log(np.maximum(inc, 1e-12))
        else:
            u = (np.maximum(inc, 1e-12)**(1-gamma) - 1) / (1-gamma)
        return float(np.sum(u * w_nodes))
    return float(golden_max(EU_of_s, a=0.0, b=1.0, tol=5e-5, max_iter=90))

def precompute_cache(kappas, theta1_grid, gamma_grid, theta2_nodes, w_nodes):
    cache = {k: np.zeros((len(theta1_grid), len(gamma_grid))) for k in kappas}
    for k in kappas:
        for i, t1 in enumerate(theta1_grid):
            for j, g in enumerate(gamma_grid):
                cache[k][i, j] = solve_s(t1, g, k, theta2_nodes, w_nodes)
    return cache

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    xi = np.searchsorted(xgrid, x) - 1; xi = np.clip(xi, 0, len(xgrid)-2)
    yi = np.searchsorted(ygrid, y_arr) - 1; yi = np.clip(yi, 0, len(ygrid)-2)
    x0, x1 = xgrid[xi], xgrid[xi+1]; y0, y1 = ygrid[yi], ygrid[yi+1]
    tx = (x - x0) / (x1 - x0 + 1e-12); ty = (y_arr - y0) / (y1 - y0 + 1e-12)
    z00 = Z[xi, yi]; z01 = Z[xi, yi+1]; z10 = Z[xi+1, yi]; z11 = Z[xi+1, yi+1]
    return (1-tx)*(1-ty)*z00 + (1-tx)*ty*z01 + tx*(1-ty)*z10 + tx*ty*z11

# -----------------------------
# Simulation per scenario (returns three dataframes)
# -----------------------------
def run_scenario(gap):
    """
    For a given μ-gap (μ2 = μ1 - gap), run the simulation across μ1_grid.
    Returns (village_compare_df, new_adopters_df, always_df).
    Note: always_df now also carries 'share_never' for plotting Row 6.
    """
    summary_rows = []
    switcher_rows = []
    always_rows = []

    for mu1 in mu1_grid:
        mu2 = max(0.001, mu1 - gap)

        # θ2 nodes for expectations (affects s*)
        a2, b2 = alpha_beta_from(mu2, sigma2_2)
        theta2_nodes = beta.ppf(x_nodes, a2, b2)
        theta2_nodes = np.clip(theta2_nodes, 1e-9, 1-1e-9)
        s_cache = precompute_cache(kappa_scenarios, theta1_grid, gamma_grid, theta2_nodes, w_nodes)

        # Worlds: θ1 ~ Beta(μ1), θ2 ~ Beta(μ2)
        a1, b1 = alpha_beta_from(mu1, sigma2_1)
        if REUSE_WORLDS:
            theta1_worlds = beta.rvs(a1, b1, size=R, random_state=rng)
            theta2_worlds = beta.rvs(a2, b2, size=R, random_state=rng)
        else:
            theta1_worlds = beta.rvs(a1, b1, size=R)
            theta2_worlds = beta.rvs(a2, b2, size=R)

        # No storage benchmark
        p1s = 1.0/(1.0 + theta1_worlds)
        inc_no = np.tile(p1s.reshape(-1, 1), (1, N))
        U_no = np.array([[crra_u(inc_no[r, i], gammas[i]) for i in range(N)] for r in range(R)])
        Y_no = inc_no.mean(axis=0)
        EU_no = U_no.mean(axis=0)
        CE_no = np.array([crra_u_inv(EU_no[i], gammas[i]) for i in range(N)])
        mean_income_no = Y_no.mean()

        # Precompute s* per κ per world
        s_all = {k: np.zeros((R, N)) for k in kappa_scenarios}
        for r in range(R):
            t1 = theta1_worlds[r]
            for kappa in kappa_scenarios:
                s_all[kappa][r, :] = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)

        # Village averages and storing shares
        for kappa in kappa_scenarios:
            Y_bar = np.zeros(N); EU_bar = np.zeros(N)
            store_count = 0
            for r in range(R):
                t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
                p1 = 1.0/(1.0 + t1); p2 = 1.0/(1.0 + t2)
                s_star = s_all[kappa][r, :]
                store_count += int(np.sum(s_star >= STORE_THRESHOLD))
                incomes = (1 - s_star)*p1 + s_star*kappa*p2
                Uvals = np.array([crra_u(incomes[i], gammas[i]) for i in range(N)])
                Y_bar += (incomes - Y_bar)/(r+1)
                EU_bar += (Uvals  - EU_bar)/(r+1)
            CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
            summary_rows.append({
                "mu1": mu1, "mu2": mu2, "gap": gap, "kappa": kappa,
                "mean_income_no_storage": mean_income_no,
                "mean_income_with_storage": Y_bar.mean(),
                "mean_gain_income": (Y_bar - Y_no).mean(),
                "mean_gain_pct": ((Y_bar - Y_no).mean()/(mean_income_no + 1e-12)),
                "mean_gain_CE": (CE_bar - CE_no).mean(),
                "share_storing": store_count/(N*R),
                "mean_s": s_all[kappa].mean(),
                "N": N, "R": R, "store_threshold": STORE_THRESHOLD
            })

        # new_adopters, Always Storers, and Never Storers between κ pairs
        for (k0, k1) in kappa_pairs:
            s0 = s_all[k0]; s1 = s_all[k1]

            idx_switch = (s0 < STORE_THRESHOLD) & (s1 >= STORE_THRESHOLD)
            share_new_adopters = idx_switch.sum()/(N*R)
            gains_switch = []

            idx_always = (s0 >= STORE_THRESHOLD) & (s1 >= STORE_THRESHOLD)
            share_always = idx_always.sum()/(N*R)
            gains_always = []

            # NEW: never storers (below threshold in both κ)
            idx_never = (s0 < STORE_THRESHOLD) & (s1 < STORE_THRESHOLD)
            share_never = idx_never.sum()/(N*R)

            for r in range(R):
                t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
                p1 = 1.0/(1.0 + t1); p2 = 1.0/(1.0 + t2)

                # new_adopters
                mask_sw = idx_switch[r, :]
                if np.any(mask_sw):
                    inc0 = (1 - s0[r, mask_sw])*p1 + s0[r, mask_sw]*k0*p2
                    inc1 = (1 - s1[r, mask_sw])*p1 + s1[r, mask_sw]*k1*p2
                    gains_switch.extend(list(inc1 - inc0))

                # Always storers
                mask_al = idx_always[r, :]
                if np.any(mask_al):
                    inc0a = (1 - s0[r, mask_al])*p1 + s0[r, mask_al]*k0*p2
                    inc1a = (1 - s1[r, mask_al])*p1 + s1[r, mask_al]*k1*p2
                    gains_always.extend(list(inc1a - inc0a))

            mean_gain_new_adopters = float(np.mean(gains_switch)) if gains_switch else 0.0
            mean_gain_always    = float(np.mean(gains_always)) if gains_always else 0.0

            switcher_rows.append({
                "mu1": mu1, "mu2": mu2, "gap": gap, "contrast": f"{k0:.2f}→{k1:.2f}",
                "share_new_adopters": share_new_adopters,
                "mean_gain_income_new_adopters": mean_gain_new_adopters,
                "N": N, "R": R, "store_threshold": STORE_THRESHOLD
            })
            # keep 'always_df' role but include share_never
            always_rows.append({
                "mu1": mu1, "mu2": mu2, "gap": gap, "contrast": f"{k0:.2f}→{k1:.2f}",
                "share_always": share_always,
                "share_never": share_never,                     # NEW
                "mean_gain_income_always": mean_gain_always,
                "N": N, "R": R, "store_threshold": STORE_THRESHOLD
            })

    village_compare_df = pd.DataFrame(summary_rows)
    new_adopters_df = pd.DataFrame(switcher_rows)
    always_df = pd.DataFrame(always_rows)   # carries share_never
    return village_compare_df, new_adopters_df, always_df

# -----------------------------
# Run all scenarios (compute first, then plot with shared y-limits)
# -----------------------------
scenario_results = []
for (label, gap) in scenarios:
    vc_df, sw_df, al_df = run_scenario(gap)
    scenario_results.append((label, gap, vc_df, sw_df, al_df))

# Compute global row-wise y-limits across all scenarios
def compute_limits(results):
    # Row 2: Mean income (no storage & with storage)
    vals_r2 = []
    for _, _, vc, _, _ in results:
        vals_r2.append(vc["mean_income_no_storage"].values)
        vals_r2.append(vc["mean_income_with_storage"].values)
    r2_vals = np.concatenate(vals_r2)
    r2_min, r2_max = float(r2_vals.min()), float(r2_vals.max())

    # Row 3: Mean income gain
    r3_vals = np.concatenate([res[2]["mean_gain_income"].values for res in results])
    r3_min, r3_max = float(r3_vals.min()), float(r3_vals.max())

    # Row 4: % mean income gain
    r4_vals = np.concatenate([100.0*res[2]["mean_gain_pct"].values for res in results])
    r4_min, r4_max = float(r4_vals.min()), float(r4_vals.max())

    # Row 5: Share of new_adopters
    r5_vals = np.concatenate([res[3]["share_new_adopters"].values for res in results])
    r5_min, r5_max = float(r5_vals.min()), float(r5_vals.max())

    # Row 6: Share of Always Storers
    r6_vals = np.concatenate([res[4]["share_always"].values for res in results])
    r6_min, r6_max = float(r6_vals.min()), float(r6_vals.max())

    # Row 7: Share of Never Storers (NEW)
    r7_vals = np.concatenate([res[4]["share_never"].values for res in results])
    r7_min, r7_max = float(r7_vals.min()), float(r7_vals.max())

    # Row 8: Income Gains for Always Storers
    r8_vals = np.concatenate([res[4]["mean_gain_income_always"].values for res in results])
    r8_min, r8_max = float(r8_vals.min()), float(r8_vals.max())

    def pad(lo, hi):
        if hi == lo:
            eps = 0.05 * (abs(hi) + 1.0)
            return lo - eps, hi + eps
        span = hi - lo
        return lo - 0.05*span, hi + 0.05*span

    return (
        pad(r2_min, r2_max),  # r2_lim
        pad(r3_min, r3_max),  # r3_lim
        pad(r4_min, r4_max),  # r4_lim
        pad(r5_min, r5_max),  # r5_lim
        pad(r6_min, r6_max),  # r6_lim (always share)
        pad(r7_min, r7_max),  # r7_lim (never share)  NEW
        pad(r8_min, r8_max),  # r8_lim (always gains)
    )

(r2_lim, r3_lim, r4_lim, r5_lim, r6_lim, r7_lim, r8_lim) = compute_limits(scenario_results)

# -----------------------------
# Plot helpers (no subplots; row-wise aligned axes)
# -----------------------------
FIGSIZE = (6, 4)
DPI = 150

def plot_mean_income_levels(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    no_storage = df.groupby("mu1", as_index=False)["mean_income_no_storage"].mean().sort_values("mu1")
    plt.plot(no_storage["mu1"], no_storage["mean_income_no_storage"], marker="o", label="No storage")
    for kappa in sorted(df["kappa"].unique()):
        d = df[df["kappa"] == kappa].sort_values("mu1")
        plt.plot(d["mu1"], d["mean_income_with_storage"], marker="o", label=f"κ={kappa}")
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Mean income (village)")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nMean Income by κ (s*≥{STORE_THRESHOLD:.0%} counts as storing)\nγ ~ 10·Beta(1,5), golden-section s*")
    plt.grid(True); plt.legend(ncols=3); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_mean_income_gain(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for kappa in sorted(df["kappa"].unique()):
        d = df[df["kappa"] == kappa].sort_values("mu1")
        plt.plot(d["mu1"], d["mean_gain_income"], marker="o", label=f"κ={kappa}")
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Mean income gain vs no-storage")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nMean Income Gain vs No Storage (s*≥{STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=3); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_pct_income_gain(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for kappa in sorted(df["kappa"].unique()):
        d = df[df["kappa"] == kappa].sort_values("mu1")
        plt.plot(d["mu1"], 100.0*d["mean_gain_pct"], marker="o", label=f"κ={kappa}")
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Mean income gain vs no-storage (%)")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\n% Mean Income Gain vs No Storage (s*≥{STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=3); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_share_new_adopters(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for contrast in df["contrast"].unique():
        d = df[df["contrast"] == contrast].sort_values("mu1")
        plt.plot(d["mu1"], d["share_new_adopters"], marker="o", label=contrast)
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Share of new_adopters")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nnew_adopters when κ increases (s* crosses {STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=2, title="κ contrast"); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_share_always(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for contrast in df["contrast"].unique():
        d = df[df["contrast"] == contrast].sort_values("mu1")
        plt.plot(d["mu1"], d["share_always"], marker="o", label=contrast)
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Share of Always Storers")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nAlways storers in both κ (s*≥{STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=2, title="κ contrast"); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

# NEW: plot share of never storers
def plot_share_never(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for contrast in df["contrast"].unique():
        d = df[df["contrast"] == contrast].sort_values("mu1")
        plt.plot(d["mu1"], d["share_never"], marker="o", label=contrast)
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Share of Never Storers")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nNever storers in both κ (s*<{STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=2, title="κ contrast"); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_gain_new_adopters(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for contrast in df["contrast"].unique():
        d = df[df["contrast"] == contrast].sort_values("mu1")
        plt.plot(d["mu1"], d["mean_gain_income_new_adopters"], marker="o", label=contrast)
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Mean income gain (new_adopters only)")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nIncome Gains for new_adopters (s* crosses {STORE_THRESHOLD:.0%})")
    plt.grid(True); plt.legend(ncols=2, title="κ contrast"); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

def plot_always_gains(df, col_title, out_path, ylim):
    plt.figure(figsize=FIGSIZE)
    for contrast in df["contrast"].unique():
        d = df[df["contrast"] == contrast].sort_values("mu1")
        plt.plot(d["mu1"], d["mean_gain_income_always"], marker="o", label=contrast)
    plt.xlabel("Mean of the Distribution of θ₁ (μ₁)")
    plt.ylabel("Mean income gain (always storers)")
    plt.ylim(*ylim)
    plt.title(f"{col_title}\nIncome Gains for Always Storers (s*≥{STORE_THRESHOLD:.0%} in both κ)")
    plt.grid(True); plt.legend(ncols=2, title="κ contrast"); plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()

# --- Add back the correct shared limits & final stitching (continuation) ---

# We need a separate row-wise limit for "Income Gains for new_adopters" (Row 7).
def compute_switch_gain_lim(results):
    vals = np.concatenate([res[3]["mean_gain_income_new_adopters"].values for res in results])
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        eps = 0.05 * (abs(hi) + 1.0); return (lo - eps, hi + eps)
    span = hi - lo
    return (lo - 0.05*span, hi + 0.05*span)

r7_switch_gain_lim = compute_switch_gain_lim(scenario_results)

# Re-run the per-scenario plotting loop (just the calls) with correct limits and panel_paths filling:

panel_paths = [[None]*4 for _ in range(8)]  # now 8 rows
scenario_dirs = []

for col_idx, (label, gap, vc_df, sw_df, al_df) in enumerate(scenario_results):
    out_dir = os.path.join(target_dir, f"scenario_{col_idx+1}")
    os.makedirs(out_dir, exist_ok=True)
    scenario_dirs.append(out_dir)

    # CSVs
    vc_path = os.path.join(out_dir, "village_compare_summary.csv")
    sw_path = os.path.join(out_dir, "new_adopters.csv")
    al_path = os.path.join(out_dir, "always_storers.csv")  # contains share_never too
    vc_df.to_csv(vc_path, index=False)
    sw_df.to_csv(sw_path, index=False)
    al_df.to_csv(al_path, index=False)

    out_r1 = os.path.join(out_dir, "row1_mean_income_levels.png")
    out_r2 = os.path.join(out_dir, "row2_mean_income_gain.png")
    out_r3 = os.path.join(out_dir, "row3_pct_mean_income_gain.png")
    out_r4 = os.path.join(out_dir, "row4_share_new_adopters.png")
    out_r5 = os.path.join(out_dir, "row5_share_always.png")
    out_r6 = os.path.join(out_dir, "row6_share_never.png")                # NEW
    out_r7 = os.path.join(out_dir, "row7_gain_new_adopters.png")
    out_r8 = os.path.join(out_dir, "row8_always_gains.png")

    plot_mean_income_levels(vc_df, label, out_r1, r2_lim)
    plot_mean_income_gain(vc_df, label, out_r2, r3_lim)
    plot_pct_income_gain(vc_df, label, out_r3, r4_lim)
    plot_share_new_adopters(sw_df, label, out_r4, r5_lim)
    plot_share_always(al_df, label, out_r5, r6_lim)
    plot_share_never(al_df, label, out_r6, r7_lim)                        # NEW
    plot_gain_new_adopters(sw_df, label, out_r7, r7_switch_gain_lim)
    plot_always_gains(al_df, label, out_r8, r8_lim)

    panel_paths[0][col_idx] = out_r1
    panel_paths[1][col_idx] = out_r2
    panel_paths[2][col_idx] = out_r3
    panel_paths[3][col_idx] = out_r4
    panel_paths[4][col_idx] = out_r5
    panel_paths[5][col_idx] = out_r6
    panel_paths[6][col_idx] = out_r7
    panel_paths[7][col_idx] = out_r8

# Stitch into a 8×4 grid image
images = [[Image.open(panel_paths[r][c]) for c in range(4)] for r in range(8)]
w, h = images[0][0].size
cols, rows = 4, 8
pad = 10
canvas_w = cols*w + (cols+1)*pad
canvas_h = rows*h + (rows+1)*pad
panel_image = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
for r in range(rows):
    for c in range(cols):
        x = pad + c*(w + pad)
        y = pad + r*(h + pad)
        panel_image.paste(images[r][c], (x, y))

final_panel_path = os.path.join(target_dir, "final_panel_8x4.png")  # updated filename
panel_image.save(final_panel_path)

final_panel_path
