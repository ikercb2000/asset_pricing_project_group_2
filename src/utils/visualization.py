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
from utils.types import plot_ptf

# Random Weights
# --------------


def random_w(n: int, alpha: float = 3.0) -> np.ndarray:
    """
    Obtains random weights following a Dirichlet distribution
    """
    return np.random.dirichlet(alpha * np.ones(n))


# Portfolio Sampler
# -----------------

def portfolio_sampler(
    returns: pd.DataFrame,
    n_portfolios: int,
    min_assets: int = 1,
    max_assets: int | None = None,
    random_state: int | None = None
) -> Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:
    """
    Generates n_portfolios random portflios, optionally around some target return
    """

    np.random.seed(random_state)

    stocks_list = list(returns.columns)
    mu_rets = ann_rets(returns)
    cov_rets = cov_mat(returns)

    if max_assets is None:
        max_assets = len(stocks_list)

    mean_var_pairs: List[List[float]] = []
    weights_list: List[np.ndarray] = []
    tickers_list: List[np.ndarray] = []

    pbar = tqdm(total=n_portfolios, desc="Sampling portfolios")

    while len(mean_var_pairs) < n_portfolios:
        # número de activos aleatorio entre min_assets y max_assets (incluido)
        k = np.random.randint(min_assets, max_assets + 1)

        selected_assets = np.random.choice(
            stocks_list, k, replace=False
        )

        # pesos en esos k activos
        # o sin alpha si prefieres tu versión básica
        w = random_w(k, alpha=3.0)

        mu_sel = mu_rets.loc[selected_assets].values
        cov_sel = cov_rets.loc[selected_assets, selected_assets].values

        ptf_ret = float(w @ mu_sel)
        ptf_var = float(w @ cov_sel @ w)

        mean_var_pairs.append([ptf_ret, ptf_var])
        weights_list.append(w)
        tickers_list.append(selected_assets)

        pbar.update(1)

    pbar.close()
    return mean_var_pairs, weights_list, tickers_list


# Mean-Variance Plot
# ------------------


def mv_plot(mv_pairs: List[List[float]], rf: float = 0.05, highlight_ptf: plot_ptf = None) -> None:
    """
    Plot Mean-Variance Frontier Graph with optional highlight portfolios
    """

    mv_pairs = np.array(mv_pairs)

    risk_free_rate = rf

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mv_pairs[:, 1]**0.5,
            y=mv_pairs[:, 0],
            marker=dict(
                color=(mv_pairs[:, 0] - risk_free_rate) /
                (mv_pairs[:, 1]**0.5),
                showscale=True,
                size=7,
                line=dict(width=1),
                colorscale="RdBu",
                colorbar=dict(title="Sharpe<br>Ratio", x=1.07),
            ),
            mode='markers',
            name="Random Portfolios"
        )
    )

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Annualised Risk (Volatility)'),
        yaxis=dict(title='Annualised Return'),
        title='Sample of Random Portfolios',
        width=850,
        height=500,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(range=[0.18, 0.32])
    fig.update_yaxes(range=[0.02, 0.27])
    fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))

    if highlight_ptf:
        for name, info in highlight_ptf.items():
            mv_pair = info.mv_pair
            color = info.color
            ret = float(mv_pair[0])
            var = float(mv_pair[1])
            vol = var**0.5

            fig.add_trace(
                go.Scatter(
                    x=[vol],
                    y=[ret],
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=12,
                        symbol="x"
                    ),
                    text=[name],
                    textposition="top center",
                    name=name
                )
            )

    fig.show()
