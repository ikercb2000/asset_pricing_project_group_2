# Packages
# --------

import pandas as pd

from typing import Dict

# Annualized Means
# ----------------


def ann_rets(returns: pd.DataFrame) -> pd.DataFrame:
    """Obtains mean vector of annualized returns"""
    return (1+returns.mean())**252-1

# Covariance Matrix
# -----------------


def cov_mat(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Obtains covariance matrix of annualized returns
    """

    return returns.cov()*252

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
