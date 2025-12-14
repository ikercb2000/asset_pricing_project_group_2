# Packages
# --------

import numpy as np
import pandas as pd
import scipy.optimize as opt

from typing import Tuple, Dict, List, Union


# Markowitz Optimization Functions
# --------------------------------


def calculate_var(weights: pd.DataFrame, cov_ret: pd.DataFrame) -> np.ndarray:
    """
    Portfolio variance objective function.
    """

    return np.dot(weights.T, np.dot(cov_ret, weights))


# Change name to optimal risky ptf

def markowitz_optrisky_ptf(mu_ret: pd.Series, cov_ret: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """
    Markowitz quadratic optimization problem tangency portfolio.
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

# 1/N Portfolio Optimiser
# -----------------------


def equally_weighted_ptf(mu_ret: pd.Series, cov_ret: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """
    Equally-weighted portfolio
    """
    tickers = list(mu_ret.index)
    n = len(tickers)

    w_vec = np.ones(n) / n

    weights_dict = {t: float(w) for t, w in zip(tickers, w_vec)}

    r = float(w_vec @ mu_ret)
    vol = float(np.sqrt(w_vec @ cov_ret.values @ w_vec))

    return weights_dict, r, vol

# Markowitz Efficient Frontier Function
# --------------------------------------


def compute_efficient_frontier(mu_rets: pd.Series, cov_rets: pd.DataFrame, n_points: int = 100, only_efficient: bool = False,
                               sup_lim: float = 0.1, inf_lim: float = -0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the efficient frontier returning the mv pairs
    """

    mu: np.ndarray = np.asarray(mu_rets.values, dtype=float).reshape(-1)
    cov: np.ndarray = np.asarray(cov_rets.values, dtype=float)

    inv_cov: np.ndarray = np.linalg.pinv(cov)

    ones: np.ndarray = np.ones_like(mu)

    A: np.ndarray = (ones @ inv_cov @ ones)
    B: np.ndarray = (ones @ inv_cov @ mu)
    C: np.ndarray = (mu  @ inv_cov @ mu)
    D: np.ndarray = (A * C - B**2)

    if D <= 0:
        raise ValueError("Covariance matrix and mean vector lead to non-positive D; "
                         "efficient frontier not well-defined.")

    mu_gmv: np.ndarray = B / A

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

    # weights
    g = (inv_cov@ones)
    h = (inv_cov@mu)
    lambda1 = (C - B * target_rets) / D
    lambda2 = (A * target_rets - B) / D

    weights = lambda1[:, None] * g[None, :] + lambda2[:, None] * h[None, :]

    return ef_pairs, weights


# Resampling Estimator Functions
# ------------------------------


def resample_inputs(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_draws: int, random_state: Union[int, None] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Resample returns from multivariate normal with data inputs
    """

    rng = np.random.default_rng(random_state)

    sim: np.ndarray = rng.multivariate_normal(
        mean=mu_ret.values, cov=cov_ret.values, size=n_draws)

    return pd.DataFrame(sim, columns=mu_ret.index)


def resampling_optimiser(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_obs: int, random_state: int, n_bootstrap: int = 100, n_points: int = 25) -> Tuple[Dict[str, float], float, float]:
    """
    Resampling optimiser as Michaud paper.
    """

    tickers: List[str] = list(cov_ret.columns)
    num_assets: int = len(tickers)
    all_weights = np.zeros((n_bootstrap, n_points, num_assets), dtype=float)

    for b in range(n_bootstrap):

        bootstrap_df: pd.DataFrame = resample_inputs(
            mu_ret=mu_ret, cov_ret=cov_ret, n_draws=n_obs, random_state=random_state + 5 * b)

        mu_bootstrap: pd.Series = bootstrap_df.mean()
        cov_bootstrap: pd.DataFrame = bootstrap_df.cov()

        _, ws_b = compute_efficient_frontier(mu_rets=mu_bootstrap, cov_rets=cov_bootstrap, n_points=n_points,
                                             only_efficient=True, sup_lim=mu_bootstrap.max())

        all_weights[b, :, :] = ws_b

    avg_weights_frontier = all_weights.mean(axis=0)
    mu_vec = mu_ret.values.astype(float)
    Sigma = cov_ret.values.astype(float)
    rets = (avg_weights_frontier @ mu_vec)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", avg_weights_frontier, Sigma, avg_weights_frontier)
                   )

    sharpe = rets / vols
    idx_best = int(np.nanargmax(sharpe))

    w_star = avg_weights_frontier[idx_best, :]
    resampled_ret: float = float(rets[idx_best])
    resampled_vol: float = float(vols[idx_best])

    avg_weights_dict: Dict[str, float] = {
        ticker: w for ticker, w in zip(tickers, w_star)
    }

    return avg_weights_dict, resampled_ret, resampled_vol

# Constrained Portfolio Optimizer
# -------------------------------


def constrained_markowitz_optimiser(mu_ret: pd.Series, cov_ret: pd.DataFrame, max_weight: float = 0.1) -> Tuple[Dict[str, float], float, float]:
    """
    Constrained minimum-variance portfolio with following restrictions:
    - fully invested (sum w_i = 1)
    - no short-selling (w_i >= 0)
    - max_weight cap on each asset (w_i <= max_weight)
    """
    tickers: List[str] = list(cov_ret.columns)
    num_assets: int = len(tickers)
    mu_vec: pd.Series = mu_ret.values
    cov_mat: pd.DataFrame = cov_ret.values

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    bounds = [(0.0, max_weight)] * num_assets

    x0 = np.repeat(1.0/num_assets, num_assets)
    res = opt.minimize(calculate_var, x0, cov_mat, method="SLSQP",
                       bounds=bounds, constraints=constraints)

    if not res.success:
        raise RuntimeError(f"Constrained optimisation failed: {res.message}")

    w_opt = res.x
    weights_dict: Dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, w_opt)
    }
    r_opt: float = float(w_opt @ mu_vec)
    vol_opt: float = float(np.sqrt(w_opt @ cov_mat @ w_opt))

    return weights_dict, r_opt, vol_opt

# Shrinkage Portfolio Functions
# -----------------------------


def _constant_correlation_target(cov_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf constant-correlation target matrix F.
    """
    s = cov_ret.values.astype(float)
    k = s.shape[0]
    std = np.sqrt(np.diag(s))
    outer_std = np.outer(std, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = s / outer_std

    np.fill_diagonal(corr, 0.0)
    r_bar = corr.sum() / (k * (k - 1))
    F = r_bar * outer_std
    np.fill_diagonal(F, np.diag(s))

    return pd.DataFrame(F, index=cov_ret.index, columns=cov_ret.columns)


def shrinkage_markowitz_optimiser(mu_ret: pd.Series, cov_ret: pd.DataFrame, n_obs: int, shrinkage: float | None = None) -> Tuple[Dict[str, float], float, float]:
    """
    Markowitz tangency portfolio using a Ledoit–Wolf-style shrinkage
    of the covariance matrix toward the constant-correlation target.
    """

    tickers: List[str] = list(cov_ret.columns)
    k: int = len(tickers)

    F = _constant_correlation_target(cov_ret)
    if shrinkage is None:
        shrinkage = k / (k + float(n_obs))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    S = cov_ret.values.astype(float)
    Fv = F.values.astype(float)
    Sigma_shrink = shrinkage * Fv + (1.0 - shrinkage) * S

    mu_vec = mu_ret.values.astype(float)
    Sigma_inv = np.linalg.inv(Sigma_shrink)
    ones = np.ones(k)

    w_unnorm = Sigma_inv @ mu_vec
    denom = float(ones @ w_unnorm)
    w = w_unnorm / denom

    weights_dict: Dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, w)
    }

    r_opt: float = float(w @ mu_vec)
    vol_opt: float = float(np.sqrt(w @ Sigma_shrink @ w))

    return weights_dict, r_opt, vol_opt
