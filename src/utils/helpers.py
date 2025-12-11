# Packages
# --------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict

# Project Modules
# ---------------

from utils.enums import FreqPrices

# Annualized Means
# ----------------


def ann_rets(returns: pd.DataFrame, frequency: FreqPrices) -> pd.DataFrame:
    """Obtains mean vector of annualized returns"""

    return ((1+returns.mean())**frequency.value)-1

# Covariance Matrix
# -----------------


def cov_mat(returns: pd.DataFrame, frequency: FreqPrices) -> pd.DataFrame:
    """
    Obtains covariance matrix of annualized returns
    """

    return (returns.cov()*frequency.value)

# Portfolio Results Print Function
# --------------------------------


def ptf_results_print(opt_ret: float, opt_vol: float, weights: Dict[str, float]) -> None:
    """
    Print the portfolio return, volatility and weights
    """

    print(f"Optimal Return: {opt_ret:.4f} | Optimal Volatility: {opt_vol:.4f}")
    print("\nWeights by Stock:\n")

    items = list(weights.items())

    for i in range(0, len(items), 5):
        row = items[i:i+5]
        row_str = " | ".join(
            f"{ticker}: {weight:.4f}" for ticker, weight in row
        )
        print(row_str)

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
