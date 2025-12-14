# Packages
# --------

import numpy as np
import matplotlib.pyplot as plt

# Colour Generation Function
# --------------------------


def generate_asset_colors(all_assets):
    """
    Generate a unique color for each asset from a fixed palette.
    Call this ONCE (e.g. before the backtest loop) and reuse.
    """
    # Combine several categorical colormaps
    base_cmaps = [
        plt.cm.tab20.colors,
        plt.cm.tab20b.colors,
        plt.cm.tab20c.colors,
    ]

    all_colors = [c for cmap in base_cmaps for c in cmap]

    if len(all_colors) < len(all_assets):
        extra_needed = len(all_assets) - len(all_colors)
        all_colors.extend(plt.cm.hsv(np.linspace(
            0, 1, extra_needed, endpoint=False)))

    return {asset: all_colors[i] for i, asset in enumerate(all_assets)}
