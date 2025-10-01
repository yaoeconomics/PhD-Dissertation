# -*- coding: utf-8 -*-
"""
Storage outcome tables at μ1 = μ2 ∈ {0.2, 0.5, 0.8}
Outputs LaTeX + CSV into a "tables" directory two levels up from this file.
"""

import os
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import beta as sp_beta

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
target_dir = os.path.join(grandparent_dir, "tables")
os.makedirs(target_dir, exist_ok=True)

TEX_OUT = os.path.join(target_dir, "gain_table_mu020508.tex")

# -----------------------------
# Global settings
# -----------------------------
rng = np.random.default_rng(314)

N = 100
R = 20000
MUS = [0.2, 0.5, 0.8]
VARIANCES = [0.02, 0.05, 0.10, 0.15]
KAPPAS = [0.80, 0.90, 0.95]
STORE_THRESHOLD = 0.01

THETA1_GRID = np.linspace(0.001, 0.999, 35)
GAMMA_GRID  = np.linspace(0.0, 10.0, 35)

nodes, weights = leggauss(12)
X_NODES = 0.5 * (nodes + 1.0)
W_NODES = 0.5 * weights

GAMMAS = rng.uniform(0.0, 10.0, size=N)
GAMMAS.sort()

# -----------------------------
# Utility functions
# -----------------------------
def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(pi)
    return (pi**(1 - gamma) - 1) / (1 - gamma)

def inv_crra_u(u_bar, gamma):
    if abs(gamma - 1.0) < 1e-12:
        return float(np.exp(u_bar))
    base = (1 - gamma) * u_bar + 1.0
    base = max(base, 1e-12)
    return float(base ** (1.0 / (1.0 - gamma)))

def alpha_beta_from_strict(mu, sigma2, eps=1e-12):
    mu = float(mu)
    if not (eps < mu < 1.0 - eps):
        return None
    max_var = mu * (1.0 - mu)
    if sigma2 >= max_var - 1e-12:
        return None
    factor = mu * (1.0 - mu) / sigma2 - 1.0
    a, b = mu * factor, (1.0 - mu) * factor
    if a <= 0.0 or b <= 0.0:
        return None
    return a, b

def golden_max(f, a=0.0, b=1.0, tol=5e-5, max_iter=100):
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
    return 0.5 * (a + b)

def precompute_s_cache(kappa, theta2_nodes):
    Z = np.zeros((len(THETA1_GRID), len(GAMMA_GRID)), dtype=float)
    for i, t1 in enumerate(THETA1_GRID):
        p1 = 1.0 / (1.0 + t1)
        for j, g in enumerate(GAMMA_GRID):
            gamma = g
            if abs(gamma) < 1e-12:
                E_inv = float(np.sum(1.0 / (1.0 + theta2_nodes) * W_NODES))
                Z[i, j] = 1.0 if (kappa * E_inv > p1) else 0.0
                continue

            def EU_of_s(s):
                inc = (1 - s) * p1 + s * kappa / (1.0 + theta2_nodes)
                if abs(gamma - 1.0) < 1e-12:
                    u = np.log(np.maximum(inc, 1e-12))
                else:
                    u = (np.maximum(inc, 1e-12) ** (1 - gamma) - 1) / (1 - gamma)
                return float(np.sum(u * W_NODES))

            Z[i, j] = float(golden_max(EU_of_s))
    return Z

def interp2_vec(xgrid, ygrid, Z, x, y_vec):
    xi = np.searchsorted(xgrid, x) - 1
    xi = np.clip(xi, 0, len(xgrid) - 2)
    yi = np.searchsorted(ygrid, y_vec) - 1
    yi = np.clip(yi, 0, len(ygrid) - 2)

    x0, x1 = xgrid[xi], xgrid[xi + 1]
    y0, y1 = ygrid[yi], ygrid[yi + 1]
    tx = (x - x0) / (x1 - x0 + 1e-12)
    ty = (y_vec - y0) / (y1 - y0 + 1e-12)

    z00 = Z[xi, yi]
    z01 = Z[xi, yi + 1]
    z10 = Z[xi + 1, yi]
    z11 = Z[xi + 1, yi + 1]

    return (1 - tx) * (1 - ty) * z00 + (1 - tx) * ty * z01 + tx * (1 - ty) * z10 + tx * ty * z11

# -----------------------------
# Main routine
# -----------------------------
def run_for_mu(mu_value):
    idx = [f"{v:.2f}" for v in VARIANCES]
    cols = [f"$\\kappa={k:.2f}$" for k in KAPPAS]
    EY = pd.DataFrame(index=idx, columns=cols, dtype=float)
    CE = pd.DataFrame(index=idx, columns=cols, dtype=float)
    S  = pd.DataFrame(index=idx, columns=cols, dtype=float)
    A  = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for v in VARIANCES:
        ab = alpha_beta_from_strict(mu_value, v)
        if ab is None:
            continue
        a, b = ab

        theta2_nodes_quad = sp_beta.ppf(X_NODES, a, b)
        theta2_nodes_quad = np.clip(theta2_nodes_quad, 1e-9, 1 - 1e-9)

        theta1_worlds = sp_beta.rvs(a, b, size=R, random_state=rng)
        theta2_worlds = sp_beta.rvs(a, b, size=R, random_state=rng)

        p1s = 1.0 / (1.0 + theta1_worlds)
        EY_no = float(np.mean(p1s))

        EU_no_by_gamma = np.array([float(np.mean(crra_u(p1s, g))) for g in GAMMAS])
        CE_no_by_gamma = np.array([inv_crra_u(EU_no_by_gamma[i], GAMMAS[i]) for i in range(N)])
        CE_no_mean = float(np.mean(CE_no_by_gamma))

        for kappa in KAPPAS:
            Z = precompute_s_cache(kappa, theta2_nodes_quad)

            Y_with_sum = 0.0
            S_sum = 0.0
            adopt_count = 0
            EU_with_by_gamma_accum = np.zeros(N, dtype=float)

            for r in range(R):
                t1, t2 = theta1_worlds[r], theta2_worlds[r]
                p1 = 1.0 / (1.0 + t1)
                p2 = 1.0 / (1.0 + t2)

                s_vec = interp2_vec(THETA1_GRID, GAMMA_GRID, Z, t1, GAMMAS)
                income_vec = (1.0 - s_vec) * p1 + s_vec * kappa * p2

                Y_with_sum += np.mean(income_vec)
                S_sum += np.mean(s_vec)
                adopt_count += int(np.mean(s_vec > STORE_THRESHOLD) * N)
                EU_with_by_gamma_accum += np.array([crra_u(iv, g) for iv, g in zip(income_vec, GAMMAS)])

            EY_with = Y_with_sum / R
            mean_s  = S_sum / R
            share_adopters = adopt_count / (R * N)

            EY_gain = 100 * (EY_with - EY_no) / (EY_no + 1e-12)

            EU_with_by_gamma = EU_with_by_gamma_accum / R
            CE_with_by_gamma = np.array([inv_crra_u(EU_with_by_gamma[i], GAMMAS[i]) for i in range(N)])
            CE_with_mean = float(np.mean(CE_with_by_gamma))
            CE_gain = 100 * (CE_with_mean - CE_no_mean) / (CE_no_mean + 1e-12)

            EY.loc[f"{v:.2f}", f"$\\kappa={kappa:.2f}$"] = EY_gain
            CE.loc[f"{v:.2f}", f"$\\kappa={kappa:.2f}$"] = CE_gain
            S.loc[f"{v:.2f}", f"$\\kappa={kappa:.2f}$"]  = mean_s
            A.loc[f"{v:.2f}", f"$\\kappa={kappa:.2f}$"]  = share_adopters

    return EY.round(2), CE.round(2), S.round(2), A.round(2)

def fmt2_no_neg_zero(x):
    """Format with 2 decimals; map tiny magnitudes to 0.00 to avoid '-0.00'."""
    if pd.isna(x):
        return ""
    v = float(x)
    if abs(v) < 0.005:   # any value that would display as 0.00 becomes +0.00
        v = 0.0
    return f"{v:.2f}"



def to_latex_triple(title, p20, p50, p80):
    lines = []
    lines.append(r"\vspace{0.35em}")
    lines.append(r"\noindent\textbf{" + title + r"}")
    lines.append(r"\vspace{0.25em}")
    lines.append(r"\begin{tabular}{l|ccc|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c|}{$\mu=0.2$} & \multicolumn{3}{c|}{$\mu=0.5$} & \multicolumn{3}{c}{$\mu=0.8$} \\")
    lines.append("Var($\\theta$) & " + " & ".join(p20.columns) + " & " +
                 " & ".join(p50.columns) + " & " + " & ".join(p80.columns) + r" \\")
    lines.append(r"\midrule")
    for idx in p20.index:
        v1 = [fmt2_no_neg_zero(p20.loc[idx, c]) for c in p20.columns]
        v2 = [fmt2_no_neg_zero(p50.loc[idx, c]) for c in p50.columns]
        v3 = [fmt2_no_neg_zero(p80.loc[idx, c]) for c in p80.columns]
        lines.append(idx + " & " + " & ".join(v1) + " & " + " & ".join(v2) + " & " + " & ".join(v3) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    EY20, CE20, S20, A20 = run_for_mu(0.2)
    EY50, CE50, S50, A50 = run_for_mu(0.5)
    EY80, CE80, S80, A80 = run_for_mu(0.8)

    latex_header = r"""\
% Auto-generated with μ ∈ {0.2,0.5,0.8}
\begin{table}[ht!]\centering
\caption{Welfare Gains of Storage when $\mu_1=\mu_2\in\{0.2,0.5,0.8\}$}
\label{tab:mu_cases_storage_outcomes}
\begin{threeparttable}
"""
    latex_footer = r"""\
\begin{tablenotes}[flushleft]
\footnotesize
\item Notes: $N=100$, $R=5000$, $\gamma\sim U[0,10]$. Prices $p=1/(1+\theta)$. $\theta_1,\theta_2\sim \text{Beta}(\alpha,\beta)$ calibrated to $(\mu,\sigma^2)$. Policy $s^*(\theta_1,\gamma)$ solved by golden-section; expectation over $\theta_2$ by 12-point Gauss–Legendre quadrature. Adopters: $s^*>0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

    body = []
    body.append(to_latex_triple("Panel A. Expected Income Gain vs No Storage (\\%)", EY20, EY50, EY80))
    body.append(to_latex_triple("Panel B. Certainty Equivalent Gain vs No Storage (\\%)", CE20, CE50, CE80))
    body.append(to_latex_triple("Panel C. Mean Storage Share", S20, S50, S80))
    body.append(to_latex_triple("Panel D. Share of Storage Adopters", A20, A50, A80))

    latex_full = latex_header + "\n\n".join(body) + "\n" + latex_footer
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(latex_full)

    print(f"LaTeX written to {TEX_OUT}")

if __name__ == "__main__":
    main()
