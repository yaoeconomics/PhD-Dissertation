# Feasible mean intervals for Beta distributions at fixed variances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

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

# Variances to consider
vars_list = [0.05, 0.10, 0.15, 0.20]

def mean_interval_for_variance(s2):
    # Solve μ(1-μ) > s2 ⇒ μ² - μ + s2 < 0
    disc = 1 - 4*s2
    if disc <= 0:
        return None  # no proper Beta with that variance
    r = sqrt(disc)
    lower = (1 - r)/2
    upper = (1 + r)/2
    return lower, upper

rows = []
for s2 in vars_list:
    lo, hi = mean_interval_for_variance(s2)
    # Example symmetric case at μ=0.5
    t = 0.25/s2 - 1
    alpha = beta = 0.5 * t
    rows.append({
        "Variance (σ²)": s2,
        "Feasible μ lower": lo,
        "Feasible μ upper": hi,
        "Example α=β at μ=0.5": alpha
    })

df = pd.DataFrame(rows)
print(df.round(6))

# Plot horizontal intervals
fig, ax = plt.subplots(figsize=(7, 3.5))

y_positions = np.arange(len(vars_list))[::-1]  # place larger variances on top
for idx, (y, s2) in enumerate(zip(y_positions, vars_list[::-1])):
    lo, hi = mean_interval_for_variance(s2)
    ax.hlines(y, lo, hi, linewidth=6, color="skyblue")
    ax.plot([lo, hi], [y, y], 'o', color="black")

    # Decide label vertical offset:
    if idx == len(vars_list) - 1:  # bottom bar
        v_offset = +0.15
        v_align = 'bottom'
    else:  # others
        v_offset = -0.15
        v_align = 'top'

    # Labels inside the bar
    ax.text(lo + 0.02, y + v_offset, f"{lo:.3f}", ha='left', va=v_align, fontsize=9, color="black")
    ax.text(hi - 0.02, y + v_offset, f"{hi:.3f}", ha='right', va=v_align, fontsize=9, color="black")

    # Variance label outside
    ax.text(1.02, y, f"σ²={s2:.2f}", va='center', fontsize=10)



ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Mean μ")
ax.set_title("Feasible mean intervals for Beta(α,β) at fixed variances")
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "parameter_range_Beta_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
