# Packages
# --------

import numpy as np
import pandas as pd

from typing import Tuple, Dict

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


def markowitz_tangency_ptf(mu_ret: pd.Series, cov_ret: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """
    Markowitz Quadratic Optimization Problem Tangency Portfolio
    """

    tickers = list(cov_ret.columns)
    num_assets: int = len(tickers)

    Sigma_inv = np.linalg.inv(cov_ret)
    ones = np.ones(num_assets)

    w_unnorm = (Sigma_inv @ mu_ret)
    denom = (ones @ w_unnorm)
    w = w_unnorm / denom

    weights_dict: Dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, w)
    }

    r_opt: float = float(w @ mu_ret)
    vol_opt: float = float(np.sqrt(w @ cov_ret @ w))

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


def resampling_optimiser(
    mu_ret: pd.Series,
    cov_ret: pd.DataFrame,
    n_obs: int,
    random_state: int,
    n_bootstrap: int = 100
) -> Tuple[Dict[str, float], float, float]:
    """
    Resampling optimiser for Optimal Portfolio with Markowitz.
    """

    tickers = list(cov_ret.columns)
    num_assets: int = len(tickers)

    all_weights = np.zeros((n_bootstrap, num_assets))

    for b in range(n_bootstrap):

        bootstrap_df = resample_inputs(
            mu_ret=mu_ret,
            cov_ret=cov_ret,
            n_draws=n_obs,
            random_state=random_state + 5*b,
        )

        mu_bootstrap: pd.Series = ann_rets(bootstrap_df, FreqPrices.YEARLY)
        cov_bootstrap: pd.DataFrame = cov_mat(bootstrap_df, FreqPrices.YEARLY)

        w_dict, _, _ = markowitz_tangency_ptf(
            mu_ret=mu_bootstrap,
            cov_ret=cov_bootstrap
        )

        all_weights[b, :] = [w_dict.get(t) for t in tickers]

    avg_weights = all_weights.mean(axis=0)

    avg_weights_dict: Dict[str, float] = {
        ticker: float(w) for ticker, w in zip(tickers, avg_weights)
    }

    w_vec = avg_weights
    resampled_ret: float = float(w_vec @ mu_ret)
    resampled_vol: float = float(np.sqrt(w_vec@cov_ret@w_vec))

    return avg_weights_dict, resampled_ret, resampled_vol
