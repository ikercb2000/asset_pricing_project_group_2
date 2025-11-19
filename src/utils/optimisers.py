# Packages
# --------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Any
from scipy.optimize import minimize

# Project Modules
# ---------------

from utils.helpers import ann_rets, cov_mat
from utils.types import plot_ptf

# Markowitz Optimization Functions
# --------------------------------


def calculate_var(weights: pd.DataFrame, cov_ret: pd.DataFrame) -> np.ndarray:
    """
    Portfolio Variance calculator
    """

    return np.dot(weights.T, np.dot(cov_ret, weights))


def markowitz_optimiser(returns: pd.DataFrame, r_min: float = 0, w_max: float = 1) -> Tuple[Dict[str, float], float, float]:
    """
    Markowitz Quadratic Optimization Problem Solver for Optimal Portfolio
    """

    mu_ret: pd.DataFrame = ann_rets(returns)
    cov_ret: pd.DataFrame = cov_mat(returns)

    tickers = list(returns.columns)
    num_assets: int = len(returns.columns)

    constraints: List[Dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: np.dot(x.T, mu_ret) - r_min},
    ]

    bounds: Tuple[Tuple[float, float]] = tuple(
        (0, w_max) for asset in range(num_assets))

    opts: Dict[str, Any] = minimize(
        fun=calculate_var,
        x0=num_assets * [1.0 / num_assets],
        args=(cov_ret,),
        method="SLSQP",
        options={"ftol": 1e-7, "maxiter": 100},
        bounds=bounds,
        constraints=constraints,
    )
    w: List[float] = opts["x"]
    weights_dict = {ticker: float(weight)
                    for ticker, weight in zip(tickers, w)}
    r_opt: float = np.dot(w, mu_ret)
    vol_opt: float = np.sqrt(calculate_var(w, cov_ret))
    return weights_dict, r_opt, vol_opt

# Resampling Optimization Functions
# ---------------------------------


def resample_inputs(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_draws: int, random_state: int | None = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Resample returns from multivariate normal with data inputs
    """

    rng = np.random.default_rng(random_state)

    sim = rng.multivariate_normal(
        mean=mu_ret.values,
        cov=cov_ret.values,
        size=n_draws
    )

    return pd.DataFrame(sim, columns=mu_ret.index)


def resampling_optimiser(returns: pd.DataFrame, random_state: int, n_bootstrap: int = 100, r_min: float = 0, w_max: float = 1) -> Tuple[Dict[str, float], float, float]:
    """
    Resampling optimiser for Optimal Portfolio with Markowitz
    """

    mu_ret: pd.DataFrame = ann_rets(returns)
    cov_ret: pd.DataFrame = cov_mat(returns)

    tickers = list(returns.columns)
    num_assets: int = len(returns.columns)

    n_draws = len(returns)

    all_weights = np.zeros((n_bootstrap, num_assets))

    for b in range(n_bootstrap):

        bootstrap_df = resample_inputs(
            mu_ret=mu_ret,
            cov_ret=cov_ret,
            n_draws=n_draws,
            random_state=random_state,
        )

        w_dict, _, _ = markowitz_optimiser(
            returns=bootstrap_df,
            r_min=r_min,
            w_max=w_max,
        )

        all_weights[b, :] = [w_dict.get(t, 0.0) for t in tickers]

    avg_weights = all_weights.mean(axis=0)

    avg_weights_dict: Dict[str, float] = {
        ticker: float(w) for ticker, w in zip(tickers, avg_weights)
    }

    w_vec = avg_weights
    resampled_ret: float = float(np.dot(w_vec, mu_ret.values))
    resampled_vol: float = float(np.sqrt(w_vec @ cov_ret.values @ w_vec))

    return avg_weights_dict, resampled_ret, resampled_vol
