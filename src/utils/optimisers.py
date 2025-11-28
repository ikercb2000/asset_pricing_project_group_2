# Packages
# --------

import numpy as np
import pandas as pd

from typing import Tuple, Dict, List

# Project Modules
# ---------------

from utils.helpers import ann_rets, cov_mat
from utils.enums import FreqPrices

# Markowitz Optimization Functions
# --------------------------------


def calculate_var(weights: pd.DataFrame, cov_ret: pd.DataFrame) -> np.ndarray:
    """
    Portfolio Variance calculator
    """

    return np.dot(weights.T, np.dot(cov_ret, weights))


# Change name to optimal risky ptf

def markowitz_tangency_ptf(mu_ret: pd.Series, cov_ret: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """
    Markowitz Quadratic Optimization Problem Tangency Portfolio
    """

    tickers: List[str] = list(cov_ret.columns)
    num_assets: int = len(tickers)

    Sigma_inv: np.ndarray = np.linalg.inv(cov_ret)
    ones: np.ndarray = np.ones(num_assets)

    w_unnorm: np.ndarray = (Sigma_inv @ mu_ret)
    denom: np.ndarray = (ones @ w_unnorm)
    w: np.ndarray = w_unnorm / denom

    weights_dict: Dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, w)
    }

    r_opt: float = float(w @ mu_ret)
    vol_opt: float = float(np.sqrt(w @ cov_ret @ w))

    return weights_dict, r_opt, vol_opt

# Markowitz Efficient Frontier Function
# --------------------------------------


def compute_efficient_frontier(mu_rets: pd.Series, cov_rets: pd.DataFrame, n_points: int = 100, only_efficient: bool = False,
                               sup_lim: float = 0.2, inf_lim: float = -0.1) -> np.ndarray:
    """
    Computes the efficient frontier returning the mv pairs
    """

    mu: np.ndarray = np.asarray(mu_rets.values, dtype=float).reshape(-1)
    cov: np.ndarray = np.asarray(cov_rets.values, dtype=float)

    inv_cov: np.ndarray = np.linalg.pinv(cov)

    ones: np.ndarray = np.ones_like(mu)

    A: np.ndarray = ones @ inv_cov @ ones
    B: np.ndarray = ones @ inv_cov @ mu
    C: np.ndarray = mu  @ inv_cov @ mu
    D: np.ndarray = A * C - B**2

    if D <= 0:
        raise ValueError("Covariance matrix and mean vector lead to non-positive D; "
                         "efficient frontier not well-defined.")

    mu_gmv: np.ndarray = B / A
    var_gmv: np.ndarray = 1.0 / A

    if only_efficient:
        ret_min: float = float(mu_gmv)
        ret_max: float = sup_lim
    else:
        # Full hyperbola
        ret_min: float = inf_lim
        ret_max: float = sup_lim

    target_rets: np.ndarray = np.linspace(ret_min, ret_max, n_points)
    target_vars: np.ndarray = (
        A * target_rets**2 - 2 * B * target_rets + C) / D

    ef_pairs: np.ndarray = np.column_stack([target_rets, target_vars])
    return ef_pairs


# Resampling Optimization Functions
# ---------------------------------


def resample_inputs(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_draws: int, random_state: int | None = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Resample returns from multivariate normal with data inputs
    """

    rng = np.random.default_rng(random_state)

    sim: np.ndarray = rng.multivariate_normal(
        mean=mu_ret.values, cov=cov_ret.values, size=n_draws)

    return pd.DataFrame(sim, columns=mu_ret.index)


def resampling_optimiser(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_obs: int,  random_state: int, n_bootstrap: int = 100) -> Tuple[Dict[str, float], float, float]:
    """
    Resampling optimiser for Optimal Portfolio with Markowitz.
    """

    tickers: List[str] = list(cov_ret.columns)
    num_assets: int = len(tickers)

    all_weights: np.ndarray = np.zeros((n_bootstrap, num_assets))

    for b in range(n_bootstrap):

        bootstrap_df = resample_inputs(
            mu_ret=mu_ret,
            cov_ret=cov_ret,
            n_draws=n_obs,
            random_state=random_state + 5*b,
        )

        mu_bootstrap: pd.Series = bootstrap_df.mean()
        cov_bootstrap: pd.DataFrame = bootstrap_df.cov()

        w_dict, _, _ = markowitz_tangency_ptf(
            mu_ret=mu_bootstrap,
            cov_ret=cov_bootstrap
        )

        all_weights[b, :] = [w_dict.get(t) for t in tickers]

    avg_weights: np.ndarray = all_weights.mean(axis=0)

    avg_weights_dict: Dict[str, float] = {
        ticker: float(w) for ticker, w in zip(tickers, avg_weights)
    }

    w_vec = avg_weights
    resampled_ret: float = float(w_vec @ mu_ret)
    resampled_vol: float = float(np.sqrt(w_vec@cov_ret@w_vec))

    return avg_weights_dict, resampled_ret, resampled_vol
