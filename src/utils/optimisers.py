# Packages
# --------

import numpy as np
import pandas as pd
import scipy.optimize as opt

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

def markowitz_optimiser(
    mu_ret: pd.Series,
    cov_ret: pd.DataFrame,
) -> Tuple[Dict[str, float], float, float]:
    """
    In-sample Markowitz optimiser (tangency portfolio).
    Used in main.py for full-sample efficient frontier.
    """
    return markowitz_tangency_ptf(mu_ret=mu_ret, cov_ret=cov_ret)

def markowitz_optimiser_oos(
    mu_ret: pd.Series,
    cov_ret: pd.DataFrame,
    n_obs: int,
) -> Tuple[Dict[str, float], float, float]:
    """
    OOS-compatible wrapper for Markowitz optimiser.
    Matches the (mu, cov, n_obs) interface used in run_oos_backtest.
    n_obs is unused, kept only for interface compatibility.
    """
    return markowitz_tangency_ptf(mu_ret=mu_ret, cov_ret=cov_ret)


def constrained_markowitz_minvar(
        mu_ret: pd.Series,
        cov_ret: pd.DataFrame,
        max_weight: float = 0.1,
) -> Tuple[Dict[str, float], float, float]:
    """
    Constrained minimum-variance portfolio:
    - fully invested (sum w_i = 1)
    - no short-selling (w_i >= 0)
    - max_weight cap on each asset (w_i <= max_weight)

    Returns (weights_dict, expected_return, volatility).
    """
    tickers = list(cov_ret.columns)
    num_assets: int = len(tickers)

    mu_vec = mu_ret.values
    cov_mat = cov_ret.values

    # Objective: portfolio variance. This is the function we want to minimise.
    def portfolio_var(w: np.ndarray) -> float:
        return float(w @ cov_mat @ w)
    
    # Constraint: Fully invested -> sum(w) = 1. This ensures the portfolio is fully invested (no borrowing or idle cash)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )

    # Bounds: 0 <= w_i <= max_weight. (Ensures no short-selling)
    bounds = [(0.0, max_weight)] * num_assets

    # Start at equal weights
    x0 = np.repeat(1.0/num_assets, num_assets)

    # Calling the optimiser (SciPy's SLSQP)
    res = opt.minimize(
        portfolio_var,
        x0,
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints,
    )

    # If there is an error, we know why
    if not res.success:
        raise RuntimeError(f"Constrained optimisation failed: {res.message}")
    
    # Extracting optimal weights
    w_opt = res.x

    # Converting optimal weights vector into a nicer mapping
    weights_dict: Dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, w_opt)
    }

    # Computing portfolio return and volatility of the optimal portfolio
    r_opt: float = float(w_opt @ mu_vec)
    vol_opt: float = float(np.sqrt(w_opt @ cov_mat @ w_opt))

    return weights_dict, r_opt, vol_opt

def constrained_markowitz_optimiser(
    mu_ret: pd.Series,
    cov_ret: pd.DataFrame,
    n_obs: int,
    max_weight: float = 0.10,
) -> Tuple[Dict[str, float], float, float]:
    """
    Wrapper to fit the (mu, cov, n_obs) optimiser interface used in run_oos_backtest.
    Uses constrained_markowitz_minvar with a given max_weight cap.
    """
    # n_obs is not used here, but kept for interface compatibility
    return constrained_markowitz_minvar(mu_ret=mu_ret, cov_ret=cov_ret, max_weight=max_weight)


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
    n_bootstrap: int = 100,
    frequency: FreqPrices | None = None,
) -> Tuple[Dict[str, float], float, float]:
    """
    Resampling optimiser for Optimal Portfolio with Markowitz.
    `frequency` is accepted for interface compatibility but not used.
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

        mu_bootstrap: pd.Series = bootstrap_df.mean()
        cov_bootstrap: pd.DataFrame = bootstrap_df.cov()

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
