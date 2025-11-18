# Packages
# --------

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np

from typing import List, Tuple
from tqdm import tqdm

# Project Modules
# ---------------

from utils.helpers import ann_rets, cov_mat

# Random Weights
# --------------


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
):

    mean_variance_pairs = []
    weights_list = []
    tickers_list = []

    mu_rets = ann_rets(returns)
    cov_rets = cov_mat(returns)

    for i in tqdm(range(n_portfolios)):
        next_i = False
        while True:
            assets = np.random.choice(
                list(returns.columns), n_assets, replace=False)

            weights = np.random.rand(n_assets)
            weights = weights/sum(weights)

            portfolio_E_Variance = 0
            portfolio_E_Return = 0
            for i in range(len(assets)):
                portfolio_E_Return += weights[i] * mu_rets.loc[assets[i]]
                for j in range(len(assets)):
                    portfolio_E_Variance += weights[i] * \
                        weights[j] * cov_rets.loc[assets[i], assets[j]]

            for R, V in mean_variance_pairs:
                if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                    next_i = True
                    break
            if next_i:
                break

            mean_variance_pairs.append(
                [portfolio_E_Return, portfolio_E_Variance])
            weights_list.append(weights)
            tickers_list.append(assets)
            break

    return mean_variance_pairs, weights_list, tickers_list


# Mean-Variance Plot
# ------------------


def mv_plot(mv_pairs: List[List[float]], rf: float = 0.05) -> None:
    mv_pairs = np.array(mv_pairs)

    risk_free_rate = rf

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mv_pairs[:, 1]**0.5, y=mv_pairs[:, 0],
                             marker=dict(color=(mv_pairs[:, 0]-risk_free_rate)/(mv_pairs[:, 1]**0.5),
                                         showscale=True,
                                         size=7,
                                         line=dict(width=1),
                                         colorscale="RdBu",
                                         colorbar=dict(title="Sharpe<br>Ratio")
                                         ),
                             mode='markers'))
    fig.update_layout(template='plotly_white',
                      xaxis=dict(title='Annualised Risk (Volatility)'),
                      yaxis=dict(title='Annualised Return'),
                      title='Sample of Random Portfolios',
                      width=850,
                      height=500)
    fig.update_xaxes(range=[0.18, 0.32])
    fig.update_yaxes(range=[0.02, 0.27])
    fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
    fig.show()
