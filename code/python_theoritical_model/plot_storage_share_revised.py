"""
Optimal first-period storage share s*(kappa, gamma) under a Beta-distributed
second-period buyer-power shock theta_2.

Revision of `plot_4x4_panel_improved.py`. MODEL unchanged: same constants, fixed
seed, CRRA objective; corner (bang-bang) solutions at gamma = 0 preserved.

Fixes carried from prior revisions: staircase removed (SMOOTH); enlarged labels;
2D contour labels de-overlapped; 2D store-all frontier; enlarged 3D panels; upright
kappa/gamma labels.

This revision fixes two flaws visible in the enlarged 3D render:
  (1) The 3D z-axis label detached and floated in the panel interior (an mplot3d
      placement artifact that worsened with box-aspect zoom).  It is replaced by a
      text2D "s*" anchored at a fixed axes fraction, which cannot float.  The z
      scale is still given by the z-ticks and by the shared colorbar.
  (2) The grey divider cut through the low-variance kappa labels.  The divider is
      now positioned by MEASURING the kappa-label extents and dropping the line
      into the clear band just below them; hspace is opened to guarantee room.

Outputs (into $OUTPUT_DIR, else ../model_figures):
    3D_formulation.png            3D surfaces
    2D_contour_formulation.png    contour alternative
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import beta
from scipy.interpolate import splprep, splev

SMOOTH = True
ZOOM3D = 1.28

# ---- model constants (identical to the original) ----
theta1 = 0.5
p1 = 1.0 / (1.0 + theta1)
delta = 1.0
s_grid = np.linspace(0, 1, 100)
gamma_grid = np.linspace(0, 5, 50)
kappa_grid = np.linspace(0.6, 1.0, 50)
num_draws = 5000

means_modified = [0.2, 0.4, 0.5]
variances = [0.02, 0.05]

# ---- styling ----
CMAP = cm.viridis
NORM = Normalize(0.0, 1.0)
BLUE = "#31507d"
ORANGE = "#a5561f"
FRONTIER = "#e15759"
HALO_T = [pe.withStroke(linewidth=3.0, foreground="0.12")]
HALO_L = [pe.withStroke(linewidth=2.6, foreground="0.15")]

plt.rcParams.update({"font.size": 15, "mathtext.fontset": "cm", "axes.titlesize": 15})


def compute_beta_params(mu, sigma2):
    factor = mu * (1.0 - mu) / sigma2 - 1.0
    return mu * factor, (1.0 - mu) * factor


def optimize_storage_share(inv, gamma, kappa):
    if gamma == 0:
        return 1.0 if delta * kappa * inv.mean() > p1 else 0.0
    income = (1 - s_grid)[:, None] * p1 + delta * s_grid[:, None] * kappa * inv[None, :]
    income = np.maximum(income, 1e-8)
    if gamma == 1:
        util = np.log(income)
    else:
        util = (income ** (1 - gamma) - 1) / (1 - gamma)
    obj = util.mean(axis=1)
    k = int(np.argmax(obj))
    if SMOOTH and 0 < k < len(s_grid) - 1:
        y0, y1, y2 = obj[k - 1], obj[k], obj[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:
            offset = np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)
            step = s_grid[1] - s_grid[0]
            return float(np.clip(s_grid[k] + offset * step, 0.0, 1.0))
    return float(s_grid[k])


pdf_params, surfaces = {}, {}
for var in variances:
    for mu in means_modified:
        a, b = compute_beta_params(mu, var)
        pdf_params[(var, mu)] = (a, b)
        theta2 = beta.rvs(a, b, size=num_draws, random_state=12345)
        inv = 1.0 / (1.0 + theta2)
        Z = np.empty((len(gamma_grid), len(kappa_grid)))
        for i, g in enumerate(gamma_grid):
            for j, k in enumerate(kappa_grid):
                Z[i, j] = optimize_storage_share(inv, g, k)
        surfaces[(var, mu)] = Z

X, Y = np.meshgrid(kappa_grid, gamma_grid)
vlow, vhigh = variances[0], variances[1]


def draw_pdf(ax, mu, var, show_ylabel):
    a, b = pdf_params[(var, mu)]
    xg = np.linspace(1e-3, 1 - 1e-3, 1000)
    yg = beta.pdf(xg, a, b)
    ax.plot(xg, yg, color=BLUE, lw=2.0)
    ax.fill_between(xg, yg, color="#4C72B0", alpha=0.22)
    ax.axvline(mu, color="0.35", lw=1.2, ls=":")
    ax.set_xlim(0, 1)
    cap = np.percentile(yg, 99) * 1.15
    ax.set_ylim(0, max(cap, np.nanmax(yg[xg > 0.05]) * 1.1))
    ax.set_title(fr"$\mu={mu},\ \sigma^2={var}$"
                 fr"    ($\alpha={a:.2f},\ \beta={b:.2f}$)", fontsize=20, pad=5)
    ax.set_xlabel(r"$\theta_2$  (second-period buyer power)", fontsize=20, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("density", fontsize=16, labelpad=4)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


# ======================================================================
# FIGURE 1 -- 3D surfaces
# ======================================================================
AX3D, TICK = 24, 16
LPAD_X, LPAD_Y = 9, 11        # reduced pad so kappa/gamma don't bleed far off-cell

fig = plt.figure(figsize=(18, 17.2))
gs = gridspec.GridSpec(4, 3, height_ratios=[0.78, 4.5, 4.5, 0.78],
                       hspace=0.52, wspace=0.16,           # hspace opened for the divider band; wspace widened to separate columns
                       left=0.098, right=0.985, top=0.940, bottom=0.105)

low_surf = []                 # row-2 (low-variance) 3D axes, for divider measurement


def draw_surface(ax, Z, is_low):
    ax.plot_surface(X, Y, Z, cmap=CMAP, norm=NORM,
                    rcount=len(gamma_grid), ccount=len(kappa_grid),
                    linewidth=0, edgecolor="none", antialiased=True, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=12, cstride=12, color="k", linewidth=0.2, alpha=0.12)
    # kappa / gamma upright; NO built-in z-label (it floats) -> use text2D instead
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel(r"$\kappa$", fontsize=AX3D, labelpad=LPAD_X, fontweight="bold", rotation=0)
    ax.set_ylabel(r"$\gamma$", fontsize=AX3D, labelpad=LPAD_Y, fontweight="bold", rotation=0)
    ax.set_zlabel("")
    ax.text2D(0.05, 0.70, r"$s^{*}$", transform=ax.transAxes, fontsize=AX3D,
              fontweight="bold", ha="left", va="center", rotation=0)
    ax.set_zlim(0, 1)
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
    if is_low:
        low_surf.append(ax)


for i, mu in enumerate(means_modified):
    draw_pdf(fig.add_subplot(gs[0, i]), mu, vlow, show_ylabel=(i == 0))
    draw_surface(fig.add_subplot(gs[1, i], projection="3d"), surfaces[(vlow, mu)], True)
    draw_surface(fig.add_subplot(gs[2, i], projection="3d"), surfaces[(vhigh, mu)], False)
    draw_pdf(fig.add_subplot(gs[3, i]), mu, vhigh, show_ylabel=(i == 0))

# ---- divider placed by MEASURING the low-variance kappa labels ----
fig.canvas.draw()
R = fig.canvas.get_renderer()
Hpx = fig.bbox.height
kappa_bottom_frac = min(ax.xaxis.label.get_window_extent(R).y0 for ax in low_surf) / Hpx
low_cell = gs[1, 0].get_position(fig)
high_cell = gs[2, 0].get_position(fig)
y_div = kappa_bottom_frac - 0.012                      # just below the kappa labels
y_div = max(y_div, high_cell.y1 + 0.010)               # but keep clear of row-3 top
fig.add_artist(Line2D([0.05, 0.985], [y_div, y_div], transform=fig.transFigure,
                      color="0.55", ls="--", lw=1.7, zorder=5))

# variance block labels, centred on each surface row
y_low = (low_cell.y0 + low_cell.y1) / 2.0
y_high = (high_cell.y0 + high_cell.y1) / 2.0
fig.text(0.012, y_low, "LOW variance\n" + fr"$\sigma^2={vlow}$", rotation=90,
         va="center", ha="center", fontsize=20, fontweight="bold", color=BLUE)
fig.text(0.012, y_high, "HIGH variance\n" + fr"$\sigma^2={vhigh}$", rotation=90,
         va="center", ha="center", fontsize=20, fontweight="bold", color=ORANGE)

cax = fig.add_axes([0.32, 0.045, 0.38, 0.016])
sm = cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=16)
cb.set_label(r"optimal first-period storage share  $s^{*}$", fontsize=19, labelpad=8)

fig.suptitle(r"Optimal storage share $s^{*}$ vs. risk aversion $\gamma$ and storage efficiency $\kappa$",
             fontsize=25, y=0.985, fontweight="bold")


# ======================================================================
# FIGURE 2 -- 2D filled-contour alternative (unchanged)
# ======================================================================
fig2 = plt.figure(figsize=(16, 9.6))
gs2 = gridspec.GridSpec(2, 3, hspace=0.34, wspace=0.16,
                        left=0.09, right=0.895, top=0.83, bottom=0.115)
fill_levels = np.linspace(0, 1, 21)
line_levels = [0.25, 0.5, 0.75]
stagger_g = {0.25: 3.9, 0.5: 2.4, 0.75: 0.9}
lab_fmt = {0.25: "0.25", 0.5: "0.50", 0.75: "0.75"}

_KA, _KB, _GB = kappa_grid[0], kappa_grid[-1], gamma_grid[-1]


def draw_smooth_frontier(ax, Xg, Yg, Z, level=0.999, color=FRONTIER, lw=2.6):
    """
    The store-all frontier is the boundary of the s*=1 plateau (a corner
    solution).  Contouring it directly gives a polyline that hugs the flat
    plateau rim on the coarse 50x50 grid, which reads as jagged.  We take the
    contour's crossing points and fit a smoothing spline through them (in
    normalized coordinates), so the drawn curve is smooth while still tracking
    the computed frontier (kappa span preserved to a fraction of a grid cell).
    The plateau is never blurred, so the frontier is preserved even where the
    store-all region is a thin sliver (high-mu panels).
    """
    ft = Figure(); at = ft.subplots()
    segs = list(at.contour(Xg, Yg, Z, levels=[level]).allsegs[0])
    for seg in segs:
        if len(seg) >= 6:
            kn = (seg[:, 0] - _KA) / (_KB - _KA)
            gn = seg[:, 1] / _GB
            try:
                tck, _ = splprep([kn, gn], s=0.0015, k=3)
                uu = np.linspace(0, 1, 300)
                xn, yn = splev(uu, tck)
                xs = xn * (_KB - _KA) + _KA       # no clipping: axes clip the display,
                ys = yn * _GB                      # so tiny overshoot never causes a kink
            except Exception:
                xs, ys = seg[:, 0], seg[:, 1]
        else:
            xs, ys = seg[:, 0], seg[:, 1]
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4)


def label_points(Z):
    pts = []
    for lev in line_levels:
        gi = int(np.argmin(np.abs(gamma_grid - stagger_g[lev])))
        row = Z[gi, :]
        if row.max() <= lev or row.min() >= lev:
            continue
        kap = float(np.interp(lev, row, kappa_grid))
        pts.append((min(max(kap, 0.62), 0.985), float(gamma_grid[gi])))
    return pts


for r, var in enumerate([vlow, vhigh]):
    for c, mu in enumerate(means_modified):
        ax = fig2.add_subplot(gs2[r, c])
        Z = surfaces[(var, mu)]
        ax.contourf(X, Y, Z, levels=fill_levels, cmap=CMAP, norm=NORM)
        cs = ax.contour(X, Y, Z, levels=line_levels, colors="white", linewidths=1.5)
        try:
            cs.set_path_effects(HALO_L)
        except Exception:
            pass
        draw_smooth_frontier(ax, X, Y, Z)
        pts = label_points(Z)
        if pts:
            lbls = ax.clabel(cs, levels=line_levels, inline=True, fontsize=13, fmt=lab_fmt, manual=pts)
            for t in lbls:
                t.set_path_effects(HALO_T)
        a, b = pdf_params[(var, mu)]
        ax.set_title(fr"$\mu={mu},\ \sigma^2={var}$   ($\alpha={a:.2f},\ \beta={b:.2f}$)",
                     fontsize=20, pad=7)
        ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.tick_params(labelsize=15)
        if r == 1:
            ax.set_xlabel(r"storage efficiency  $\kappa$", fontsize=19, labelpad=4)
        if c == 0:
            ax.set_ylabel(r"risk aversion  $\gamma$", fontsize=19, labelpad=4)

fig2.text(0.020, 0.645, "LOW variance", rotation=90, va="center", ha="center",
          fontsize=18, fontweight="bold", color=BLUE)
fig2.text(0.020, 0.235, "HIGH variance", rotation=90, va="center", ha="center",
          fontsize=18, fontweight="bold", color=ORANGE)

cax2 = fig2.add_axes([0.910, 0.115, 0.018, 0.715])
sm2 = cm.ScalarMappable(cmap=CMAP, norm=NORM); sm2.set_array([])
cb2 = fig2.colorbar(sm2, cax=cax2)
cb2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb2.ax.tick_params(labelsize=16)
cb2.set_label(r"optimal first-period storage share  $s^{*}$", fontsize=18, labelpad=8)

fig2.suptitle(r"Optimal storage share $s^{*}(\kappa,\gamma)$"
              "\n"
              r"white curves: iso-storage contours    "
              + r"$\bf{red\ curve}$: store-all frontier ($s^{*}\!\to\!1$)",
              fontsize=20, y=0.965, fontweight="bold")

# ---- save ----
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.environ.get("OUTPUT_DIR", os.path.join(grandparent_dir, "model_figures"))
os.makedirs(target_dir, exist_ok=True)
fig.savefig(os.path.join(target_dir, "3D_formulation_revision.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(target_dir, "2D_contour_formulation_revision.png"), dpi=300, bbox_inches="tight")
plt.show()
