# Packages
# --------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Union, Any
from pathlib import Path

# Project Modules
# ---------------

from utils.types import plot_ptf
from utils.helpers import generate_asset_colors
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

def mv_plot(mv_pairs: List[List[float]], save_path: str, show_plot: bool = True, rf: float = 0.05, highlight_ptf: Dict[str, plot_ptf] = None,
            ef_pairs: Union[np.ndarray, None] = None) -> None:
    """
    Plot Mean-Variance Frontier Graph
    """
    mv_pairs = np.asarray(mv_pairs, dtype=float)
    vols = np.sqrt(mv_pairs[:, 1])
    rets = mv_pairs[:, 0]

    sharpe = np.zeros_like(rets)
    nonzero = vols > 0
    sharpe[nonzero] = (rets[nonzero] - rf) / vols[nonzero]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    sc = ax.scatter(vols, rets, c=sharpe, cmap="RdBu", s=30,
                    edgecolors="black", linewidths=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", rotation=90)

    if ef_pairs is not None:
        ef_pairs = np.asarray(ef_pairs, dtype=float)
        ef_vol = np.sqrt(ef_pairs[:, 1])
        ef_ret = ef_pairs[:, 0]
        ax.plot(ef_vol, ef_ret, linewidth=2.5, label="Efficient Frontier")

    if highlight_ptf:
        for name, info in highlight_ptf.items():
            mv_pair: Tuple[float, float] = info.mv_pair
            color: str = info.color
            ret_h = float(mv_pair[0])
            vol_h = float(mv_pair[1]) ** 0.5

            ax.scatter([vol_h], [ret_h], marker="x", s=80,
                       color=color, label=name, zorder=5)
            ax.text(vol_h, ret_h, f" {name}", color=color,
                    fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Monthly Risk (Volatility)")
    ax.set_ylabel("Monthly Return")
    ax.set_title("Sample of Random Portfolios")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="lower right", frameon=True, framealpha=0.7,
                           facecolor="white", edgecolor="black", fontsize=9)
        legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    save_path = str(Path(save_path))
    fig.savefig(save_path, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# Plot Window MV Function
# -----------------------


def plot_window_mv(end: int, date: pd.Timestamp, mv_plot_dir: Union[str, Path], show_plots: bool, rf: float, global_mv_pairs, eff_front: np.ndarray,
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


def do_plot_batch(end, date, mv_plot_dir, show_plots, rf: float, mu_w: Union[np.ndarray, pd.Series],
                  cov_w: Union[np.ndarray, pd.DataFrame], n_ptfs: int, min_assets: int, max_assets: int,
                  random_state: int, window_mv_data) -> None:
    """
    Plots for batch and executes in parallel inside each thread
    """

    mv_pairs_t, _, _ = portfolio_sampler(mu_rets=mu_w, cov_rets=cov_w, n_portfolios=n_ptfs, min_assets=min_assets,
                                         max_assets=max_assets, random_state=random_state)

    eff_front, _ = compute_efficient_frontier(mu_w, cov_w, n_ptfs)

    plot_window_mv(end, date, mv_plot_dir, show_plots, rf,
                   mv_pairs_t, eff_front, window_mv_data)

# Allocation Plot Function
# ------------------------


def plot_allocation_frame(weights_df: pd.DataFrame, end: int, date: pd.Timestamp, save_dir: str | Path,
                          show_plot: bool = False, asset_colors: Dict[str, Any] | None = None) -> None:
    """
    Plot of the allocation for each asset in a portfolio with a clean legend.
    asset_colors: dict global asset -> color (precomputed once for speed).
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assets = list(weights_df.index)
    portfolios = list(weights_df.columns)

    n_ptf = len(portfolios)
    x = np.arange(1, n_ptf + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = np.zeros(n_ptf, dtype=float)
    weights_pct = weights_df.values * 100.0

    if asset_colors is None:
        asset_colors = generate_asset_colors(assets)

    for i, asset in enumerate(assets):
        w = weights_pct[i, :]
        color = asset_colors.get(asset, None)
        ax.bar(
            x,
            w,
            bottom=bottoms,
            label=asset,
            color=color,
        )
        bottoms += w

    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, n_ptf + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios, rotation=45, ha="right")
    ax.set_xlabel("Portfolio Method")
    ax.set_ylabel("Allocation (%)")
    ax.set_title(f"Portfolio Allocations | {date}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    n_assets = len(assets)
    ncols = 1 if n_assets <= 12 else 2 if n_assets <= 24 else 3
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=True,
        ncol=ncols,
        title="Assets",
        title_fontsize=9,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])

    date_str = str(date).replace(" ", "-")
    date_str = date_str.replace(":", "-")
    fname = save_dir / f"alloc_{end:04d}_{date_str}.png"
    fig.savefig(fname, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
