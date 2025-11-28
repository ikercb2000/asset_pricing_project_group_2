# Packages
# --------

import pandas as pd
import numpy as np
import plotly.graph_objects as go


from joblib import Parallel, delayed
from typing import List, Tuple, Dict
from IPython.display import Image, display
from pathlib import Path

# Project Modules
# ---------------

from utils.types import plot_ptf
from utils.optimisers import compute_efficient_frontier

# Random Weights Function
# -----------------------


def random_w_leverage(k: int, net_exposure: float = 1.0, gross_limit: float = 2.0) -> np.ndarray:
    """
    Obtains random weights that allow leverage and short-selling
    """

    w: np.ndarray = np.random.randn(k)
    w: np.ndarray = w / np.sum(w) * net_exposure

    gross: float = np.sum(np.abs(w))
    if gross > gross_limit:
        w = w / gross * gross_limit

    return w

# Portfolio Sampler Function
# --------------------------


def sample_one(mu_arr: pd.Series, cov_arr: pd.DataFrame, tickers: List[str], n_assets_total: int,
               min_assets: int, max_assets: int, net_exposure: float, gross_limit: float,
               rng_seed: int) -> Tuple[float, List[float], List[str]]:

    rng = np.random.default_rng(rng_seed)

    k: int = rng.integers(min_assets, max_assets + 1)
    idx = rng.choice(n_assets_total, size=k, replace=False)

    w: np.ndarray = random_w_leverage(k, net_exposure=net_exposure,
                                      gross_limit=gross_limit)

    mu_sel: pd.Series = mu_arr[idx]
    cov_sel: pd.DataFrame = cov_arr[np.ix_(idx, idx)]

    ret: float = float(w @ mu_sel)
    var: float = float(w @ cov_sel @ w)

    return (ret, var, w, tickers[idx])


def portfolio_sampler(mu_rets: pd.Series, cov_rets: pd.DataFrame, n_portfolios: int, min_assets: int = 1, max_assets: int = None,
                      random_state: int = 123, n_jobs: int = -1, net_exposure: float = 1.0,
                      gross_limit: float = 3.0) -> Tuple[List[Tuple[float, float]], List[float], List[str]]:

    tickers: np.ndarray = np.array(cov_rets.columns)
    mu_arr: np.ndarray = mu_rets.values
    cov_arr: np.ndarray = cov_rets.values
    n_assets_total: int = len(tickers)

    if max_assets is None:
        max_assets = n_assets_total

    seeds = np.random.SeedSequence(random_state).spawn(n_portfolios)

    results = Parallel(n_jobs=n_jobs)(delayed(sample_one)
                                      (mu_arr, cov_arr, tickers, n_assets_total, min_assets,
                                       max_assets, net_exposure, gross_limit, s.generate_state(1)[0]) for s in seeds)

    mean_var_pairs: List[Tuple[float, float]] = [
        (r, v) for (r, v, _, _) in results]
    weights_list: List[float] = [w for (_, _, w, _) in results]
    tickers_list: List[float] = [t for (_, _, _, t) in results]

    return mean_var_pairs, weights_list, tickers_list


# Mean-Variance Plot
# ------------------

def mv_plot(mv_pairs: List[List[float]], save_path: str, show_plot: bool = True, rf: float = 0.05,
            highlight_ptf: plot_ptf = None, ef_pairs: np.ndarray | None = None) -> None:
    """
    Plot Mean-Variance Frontier Graph with optional highlight portfolios
    and an optional efficient frontier line.
    """
    mv_pairs: np.ndarray = np.array(mv_pairs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mv_pairs[:, 1]**0.5, y=mv_pairs[:, 0],
                             marker=dict(color=(mv_pairs[:, 0] - rf) /
                                         (mv_pairs[:, 1]**0.5),
                                         showscale=True,
                                         size=7,
                                         line=dict(width=1),
                                         colorscale="RdBu",
                                         colorbar=dict(
                                             title="Sharpe<br>Ratio", x=1.07),
                                         ), mode='markers', name="Random Portfolios"))

    if ef_pairs is not None:

        ef_pairs: np.ndarray = np.asarray(ef_pairs)
        ef_vol: float = np.sqrt(ef_pairs[:, 1])
        ef_ret: float = ef_pairs[:, 0]

        fig.add_trace(go.Scatter(x=ef_vol, y=ef_ret, mode="lines",
                      line=dict(width=3), name="Efficient Frontier",))

    if highlight_ptf:

        for name, info in highlight_ptf.items():
            mv_pair: Tuple[float, float] = info.mv_pair
            color: str = info.color
            ret: float = float(mv_pair[0])
            vol: float = float(mv_pair[1])**0.5

            fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers+text", marker=dict(color=color, size=12, symbol="x"),
                                     text=[name], textposition="top center", name=name))

    fig.update_layout(template='plotly_white', xaxis=dict(title='Monthly Risk (Volatility)'), yaxis=dict(title='Monthly Return'),
                      title='Sample of Random Portfolios', width=850, height=500,
                      legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor="rgba(0,0,0,0.2)", borderwidth=1, font=dict(size=10), itemsizing="trace"))

    fig.write_image(save_path, scale=2)

    if show_plot:
        display(Image(filename=save_path))

# Plot Window MV Function
# -----------------------


def plot_window_mv(end: int, date: pd.Timestamp, mv_plot_dir: str | Path, show_plots: bool, rf: float, global_mv_pairs, eff_front: np.ndarray,
                   window_mv_data: Dict[str, Tuple[float, float]]) -> None:

    highlights: Dict[str, plot_ptf] = {}
    colors: List[str] = ["green", "orange", "purple", "black", "red", "blue"]

    for i, (m_name, (r_m, vol_m)) in enumerate(window_mv_data.items()):
        highlights[m_name] = plot_ptf(
            mv_pair=(r_m, vol_m**2),   # mensual
            color=colors[i % len(colors)],
        )

    date_str: str = str(date).replace(" ", "-")
    fname = Path(mv_plot_dir, f"mv_oos_{end:04d}_{date_str}.png")

    mv_plot(mv_pairs=global_mv_pairs, save_path=str(
        fname), show_plot=show_plots, rf=rf, highlight_ptf=highlights, ef_pairs=eff_front)

# Batch Ploto Function
# --------------------


def do_plot_batch(end, date, mv_plot_dir: str, show_plots, rf: float, mu_w: np.ndarray | pd.Series, cov_w, n_ptfs, min_assets, max_assets,
                  random_state, window_mv_data) -> None:
    """
    Plots for batch and executes in parallel inside each thread
    """
    mv_pairs_t, _, _ = portfolio_sampler(mu_rets=mu_w, cov_rets=cov_w, n_portfolios=n_ptfs, min_assets=min_assets,
                                         max_assets=max_assets, random_state=random_state)

    eff_front = compute_efficient_frontier(mu_w, cov_w, n_ptfs)

    plot_window_mv(end, date, mv_plot_dir, show_plots, rf,
                   mv_pairs_t, eff_front, window_mv_data)
