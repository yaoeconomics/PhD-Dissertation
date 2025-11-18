# Re-run the full integration cell after the transient kernel reset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss
from io import BytesIO
from PIL import Image


import os

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



rng = np.random.default_rng(304)

# Config
mu_grid = np.arange(0.15, 0.86, 0.05)
sigma2 = 0.10
N = 100
gammas = rng.uniform(0, 10, size=N); gammas.sort()
# kappa_pairs = [(0.80, 0.85), (0.80, 0.90), (0.80, 0.95)]
kappa_pairs = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
R = 5000
nodes, weights = leggauss(12)
x_nodes = 0.5*(nodes+1); w_nodes = 0.5*weights

def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma-1.0) < 1e-12: return np.log(pi)
    return (pi**(1-gamma)-1)/(1-gamma)

def crra_u_inv(u, gamma):
    if abs(gamma-1.0) < 1e-12: return np.exp(u)
    return np.maximum(u*(1-gamma)+1, 1e-12)**(1/(1-gamma))

def alpha_beta_from(mu, sigma2):
    factor = mu*(1-mu)/sigma2 - 1
    return mu*factor, (1-mu)*factor

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes, E_inv1p_theta2):
    p1 = 1.0/(1.0+theta1)
    if abs(gamma) < 1e-12:
        return 1.0 if (kappa*E_inv1p_theta2 > p1) else 0.0
    s_grid = np.linspace(0,1,33)
    EU = []
    for s in s_grid:
        incomes = (1-s)*p1 + s*kappa/(1.0+theta2_nodes)
        EU.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    s0 = s_grid[int(np.argmax(EU))]
    lo, hi = max(0.0, s0-0.1), min(1.0, s0+0.1)
    s_fine = np.linspace(lo, hi, 33)
    EUf = []
    for s in s_fine:
        incomes = (1-s)*p1 + s*kappa/(1.0+theta2_nodes)
        EUf.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    return float(s_fine[int(np.argmax(EUf))])

def precompute_cache(kappas, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2):
    cache = {k: np.zeros((len(theta1_grid), len(gamma_grid))) for k in kappas}
    for k in kappas:
        for i, t1 in enumerate(theta1_grid):
            for j, g in enumerate(gamma_grid):
                cache[k][i,j] = solve_s(t1, g, k, theta2_nodes, w_nodes, E_inv1p_theta2)
    return cache

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    xi = np.searchsorted(xgrid, x) - 1
    xi = np.clip(xi, 0, len(xgrid)-2)
    x0, x1 = xgrid[xi], xgrid[xi+1]
    tx = (x - x0) / (x1 - x0 + 1e-12)
    yi = np.searchsorted(ygrid, y_arr) - 1
    yi = np.clip(yi, 0, len(ygrid)-2)
    y0 = ygrid[yi]; y1 = ygrid[yi+1]
    ty = (y_arr - y0) / (y1 - y0 + 1e-12)
    z00 = Z[xi, yi]; z01 = Z[xi, yi+1]; z10 = Z[xi+1, yi]; z11 = Z[xi+1, yi+1]
    return (1-tx)*(1-ty)*z00 + (1-tx)*ty*z01 + tx*(1-ty)*z10 + tx*ty*z11

def evaluate_kappa(mu, kappas, gammas, R):
    alpha, beta_param = alpha_beta_from(mu, sigma2)
    theta1_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_nodes = beta.ppf(x_nodes, alpha, beta_param)
    theta2_nodes = np.clip(theta2_nodes, 1e-9, 1-1e-9)
    E_inv1p_theta2 = float(np.sum(1.0/(1.0+theta2_nodes)*w_nodes))
    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid = np.linspace(0.0, 10.0, 15)
    s_cache = precompute_cache(kappas, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2)
    out = {}
    for kappa in kappas:
        s_bar = np.zeros(N); Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0/(1.0+t1); p2 = 1.0/(1.0+t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            incomes = (1 - s_star)*p1 + s_star*kappa*p2
            Uvals = np.array([crra_u(incomes[i], gammas[i]) for i in range(N)])
            s_bar += (s_star - s_bar)/(r+1)
            Y_bar += (incomes - Y_bar)/(r+1)
            EU_bar += (Uvals - EU_bar)/(r+1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RP_bar = Y_bar - CE_bar
        out[kappa] = {"mean_Y": float(Y_bar.mean()), "mean_CE": float(CE_bar.mean()),
                      "mean_RP": float(RP_bar.mean()), "mean_s": float(s_bar.mean())}
    return out

# Aggregate gains for line charts
summary_rows, gain_rows = [], []
for mu in mu_grid:
    kappas_needed = sorted(set([k for pair in kappa_pairs for k in pair]))
    evals = evaluate_kappa(mu, kappas_needed, gammas, R)
    for k in kappas_needed:
        summary_rows.append({"mu": mu, "kappa": k,
                             "mean_income": evals[k]["mean_Y"],
                             "mean_CE": evals[k]["mean_CE"],
                             "mean_RP": evals[k]["mean_RP"],
                             "mean_s": evals[k]["mean_s"]})
    for k0, k1 in kappa_pairs:
        dY = evals[k1]["mean_Y"] - evals[k0]["mean_Y"]
        dCE = evals[k1]["mean_CE"] - evals[k0]["mean_CE"]
        dRP = evals[k1]["mean_RP"] - evals[k0]["mean_RP"]
        gain_rows.append({"mu": mu, "contrast": f"{k0:.2f}→{k1:.2f}",
                          "ΔE[π]": dY, "ΔCE": dCE, "ΔRP": dRP})
summary_df = pd.DataFrame(summary_rows)
gains_df = pd.DataFrame(gain_rows)

# --- NEW: compute a global y-range for column-1 line charts ---
contrasts = [f"{k0:.2f}→{k1:.2f}" for k0, k1 in kappa_pairs]
vals_for_ylim = []
for c in contrasts:
    d = gains_df[gains_df["contrast"] == c]
    vals_for_ylim.append(d["ΔCE"].values)
    vals_for_ylim.append(d["ΔE[π]"].values)
    vals_for_ylim.append((-d["ΔRP"]).values)  # insurance gain is plotted as -ΔRP
all_vals = np.concatenate(vals_for_ylim)
# Add a small 5% padding for readability
vmin, vmax = all_vals.min(), all_vals.max()
pad = 0.05*(vmax - vmin if vmax > vmin else (abs(vmax) + 1e-12))
common_ylim = (vmin - pad, vmax + pad)


def render_line_tile(df, contrast, ylim=None):
    d = df[df["contrast"] == contrast].sort_values("mu")
    fig = plt.figure(figsize=(4, 2.6))
    ax = fig.add_subplot(111)
    ax.plot(d["mu"], d["ΔCE"], linewidth=2, label="ΔCE")
    ax.plot(d["mu"], d["ΔE[π]"], linewidth=2, label="ΔE[π]")
    ax.plot(d["mu"], -d["ΔRP"], linewidth=2, label="−ΔRP (insurance gain)")

    ax.set_xlabel("mean buyer power μ", fontsize=14)
    ax.set_ylabel("Gain (price units)", fontsize=14)
    ax.set_title(f"κ {contrast}: Gains vs μ", fontsize=14)

    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=10)
    ax.grid(True)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


line_tiles = []
for contrast in contrasts:
    line_tiles.append(render_line_tile(gains_df, contrast, ylim=common_ylim))

# Decile heatmaps
deciles = pd.qcut(gammas, 10, labels=False)

records = []
for mu in mu_grid:
    alpha, beta_param = alpha_beta_from(mu, sigma2)
    theta1_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_nodes = beta.ppf(x_nodes, alpha, beta_param)
    theta2_nodes = np.clip(theta2_nodes, 1e-9, 1 - 1e-9)
    E_inv1p_theta2 = float(np.sum(1.0 / (1.0 + theta2_nodes) * w_nodes))

    kappas_needed = sorted(set([k for pair in kappa_pairs for k in pair]))
    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid = np.linspace(0.0, 10.0, 15)
    s_cache = precompute_cache(kappas_needed, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2)

    level_by_kappa = {}
    for kappa in kappas_needed:
        Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0 / (1.0 + t1); p2 = 1.0 / (1.0 + t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            incomes = (1 - s_star) * p1 + s_star * kappa * p2
            Uvals = np.array([crra_u(incomes[i], gammas[i]) for i in range(N)])
            Y_bar += (incomes - Y_bar) / (r + 1)
            EU_bar += (Uvals - EU_bar) / (r + 1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RP_bar = Y_bar - CE_bar
        df = pd.DataFrame({"gamma": gammas, "decile": deciles, "Y": Y_bar, "CE": CE_bar, "RP": RP_bar})
        level_by_kappa[kappa] = df

    for k0, k1 in kappa_pairs:
        df0 = level_by_kappa[k0]
        df1 = level_by_kappa[k1]
        d = df1.merge(df0, on=["gamma", "decile"], suffixes=("_1", "_0"))
        d["dY"] = d["Y_1"] - d["Y_0"]
        d["dCE"] = d["CE_1"] - d["CE_0"]
        d["dRP"] = d["RP_1"] - d["RP_0"]
        dec = d.groupby("decile").agg(
            mean_dY=("dY", "mean"),
            mean_dCE=("dCE", "mean"),
            mean_dRP=("dRP", "mean"),
            gamma_lo=("gamma", "min"),
            gamma_hi=("gamma", "max")
        ).reset_index()
        dec["mu"] = mu; dec["contrast"] = f"{k0:.2f}→{k1:.2f}"
        records.append(dec)

panel = pd.concat(records, ignore_index=True)


def render_heatmap_image_shared(panel, contrast, metric, title_suffix, mu_grid, vmin, vmax):
    dfc = panel[panel["contrast"] == contrast].copy()
    df_pivot = dfc.pivot(index="decile", columns="mu", values=metric).sort_index()
    vtitle = title_suffix
    if metric == "mean_dRP":
        df_pivot = -df_pivot
        vtitle = f"{title_suffix}"

    fig = plt.figure(figsize=(4, 2.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(df_pivot.values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.4/10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_yticks(np.arange(0, 10))
    ax.set_yticklabels([f"{i+1}" for i in range(10)], fontsize=14)

    tick_idx = np.arange(0, len(mu_grid), 3)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{mu_grid[i]:.2f}" for i in tick_idx],
                       rotation=45, ha="right", fontsize=14)

    ax.set_xlabel("mean buyer power μ", fontsize=14)
    ax.set_ylabel("γ decile", fontsize=14)
    ax.set_title(f"{contrast}: {vtitle}", fontsize=14)

    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


metrics = [("mean_dCE", "ΔCE"),
           ("mean_dY", "ΔE[π]"),
           ("mean_dRP", "Insurance gain")]

metric_ranges = {}
for metric, _ in metrics:
    vals = []
    for contrast in contrasts:
        dfc = panel[panel["contrast"] == contrast].copy()
        piv = dfc.pivot(index="decile", columns="mu", values=metric).sort_index()
        if metric == "mean_dRP":
            piv = -piv
        vals.append(piv.values)
    allv = np.concatenate([v.ravel() for v in vals])
    metric_ranges[metric] = (np.nanmin(allv), np.nanmax(allv))

heatmap_tiles = {contrast: [] for contrast in contrasts}
for contrast in contrasts:
    for metric, title in metrics:
        vmin, vmax = metric_ranges[metric]
        img = render_heatmap_image_shared(panel, contrast, metric, title, mu_grid, vmin, vmax)
        heatmap_tiles[contrast].append(img)

# Compose 3×4 grid
all_tiles = line_tiles + [im for row in heatmap_tiles.values() for im in row]
cell_w = max(im.size[0] for im in all_tiles)
cell_h = max(im.size[1] for im in all_tiles)

rows, cols = 4, 4
canvas_w = cols * cell_w
canvas_h = rows * cell_h
grid_img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

for r, contrast in enumerate(contrasts):
    # col 1 line
    tile = line_tiles[r]
    x0 = 0 * cell_w + (cell_w - tile.size[0]) // 2
    y0 = r * cell_h + (cell_h - tile.size[1]) // 2
    grid_img.paste(tile, (x0, y0))
    # cols 2–4 heatmaps
    for c in range(3):
        tile = heatmap_tiles[contrast][c]
        x0 = (c + 1) * cell_w + (cell_w - tile.size[0]) // 2
        y0 = r * cell_h + (cell_h - tile.size[1]) // 2
        grid_img.paste(tile, (x0, y0))

png_path = os.path.join(target_dir, "storage_subsidy_gain_heatmap.png")
grid_img.save(png_path)

print((png_path))


import matplotlib.pyplot as plt

# At the end of your script, after saving grid_img
plt.figure(figsize=(16, 12))
plt.imshow(grid_img)
plt.axis("off")
plt.tight_layout()
plt.show()
