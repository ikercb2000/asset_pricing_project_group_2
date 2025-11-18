# Packages
# --------

import pandas as pd
import numpy as np

from typing import List, Tuple

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

# Covariance Matrix
# -----------------


def random_w(n: int) -> pd.DataFrame:
    """
    Obtains normalized random weights for a portfolio
    """
    unnorm_weights = np.random.rand(n)
    return unnorm_weights/sum(unnorm_weights)


# Portfolio Sampler
# -----------------

def portfolio_sampler(
    returns: pd.DataFrame,
    n_assets: int,
    n_portfolios: int,
    just_eff: bool = False
) -> Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:
    """
    Generates n_portfolios random portflios. Allows to just plot efficient frontier.
    """

    stocks_list = list(returns.columns)

    mu_rets = ann_rets(returns)
    cov_rets = cov_mat(returns)

    mean_var_pairs: List[List[float]] = []
    weights_list: List[np.ndarray] = []
    tickers_list: List[np.ndarray] = []

    for _ in range(n_portfolios):
        selected_assets = np.random.choice(
            stocks_list, n_assets, replace=False)

        weights = random_w(n_assets)
        w = np.asarray(weights)

        mu_sel = mu_rets.loc[selected_assets].values
        cov_sel = cov_rets.loc[selected_assets, selected_assets].values

        ptf_excess_ret = float(w @ mu_sel)
        ptf_excess_var = float(w @ cov_sel @ w)

        if just_eff:
            dominated = False
            for R, V in mean_var_pairs:
                if (R > ptf_excess_ret) and (V < ptf_excess_var):
                    dominated = True
                    break

            if dominated:
                continue

        mean_var_pairs.append([ptf_excess_ret, ptf_excess_var])
        weights_list.append(w)
        tickers_list.append(selected_assets)

    return mean_var_pairs, weights_list, tickers_list

# Portfolio Sampler for Efficent Frontier
# ---------------------------------------


def portfolio_sampler_eff(
    returns: pd.DataFrame,
    n_assets: int,
    n_portfolios: int,
) -> Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:

    stocks_list = list(returns.columns)
    mu_rets = ann_rets(returns)
    cov_rets = cov_mat(returns)

    mean_var_pairs: List[List[float]] = []
    weights_list: List[np.ndarray] = []
    tickers_list: List[np.ndarray] = []

    count = 0

    while count < n_portfolios:

        selected_assets = np.random.choice(
            stocks_list, n_assets, replace=False)

        w = random_w(n_assets)

        mu_sel = mu_rets.loc[selected_assets].values
        cov_sel = cov_rets.loc[selected_assets, selected_assets].values

        ptf_ret = float(w @ mu_sel)
        ptf_var = float(w @ cov_sel @ w)

        dominated = False
        for R, V in mean_var_pairs:
            if (R > ptf_ret) and (V < ptf_var):
                dominated = True
                break
        if dominated:
            continue

        # Portfolio válido → guardarlo
        mean_var_pairs.append([ptf_ret, ptf_var])
        weights_list.append(w)
        tickers_list.append(selected_assets)
        count += 1

    return mean_var_pairs, weights_list, tickers_list
