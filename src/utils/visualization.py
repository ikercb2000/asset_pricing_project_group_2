# Packages
# --------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Union, Any, Sequence
from pathlib import Path
from PIL import Image

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
    """
    Generate a single randomly sampled portfolio and compute its return and variance.
    """

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
    """
    Sample different portfolios inside the mean-variance frontier.
    """

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

def mv_plot(mv_pairs: List[List[float]], save_path: str, date: pd.Timestamp, show_plot: bool = True, highlight_ptf: Dict[str, plot_ptf] = None,
            ef_pairs: Union[np.ndarray, None] = None) -> None:
    """
    Plot Mean-Variance Frontier Graph
    """
    mv_pairs = np.asarray(mv_pairs, dtype=float)
    vols = np.sqrt(mv_pairs[:, 1])
    rets = mv_pairs[:, 0]

    sharpe = np.zeros_like(rets)
    nonzero = vols > 0
    sharpe[nonzero] = rets[nonzero] / vols[nonzero]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    sc = ax.scatter(vols, rets, c=sharpe, cmap="RdBu", s=30,
                    edgecolors="black", linewidths=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe (excess)", rotation=90)

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
    ax.set_ylabel("Monthly Excess Return")
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    ax.set_title(f"Sample of Random Portfolios | {date_str}")

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


def plot_window_mv(end: int, date: pd.Timestamp, mv_plot_dir: Union[str, Path], show_plots: bool, global_mv_pairs, eff_front: np.ndarray,
                   window_mv_data: Dict[str, Tuple[float, float]]) -> None:
    """
    Plot and save the mean–variance representation for a single rolling-window period.
    """

    highlights: Dict[str, plot_ptf] = {}
    colors: List[str] = ["green", "orange", "purple", "black", "red", "blue"]

    for i, (m_name, (r_m, vol_m)) in enumerate(window_mv_data.items()):
        highlights[m_name] = plot_ptf(
            mv_pair=(r_m, vol_m**2),
            color=colors[i % len(colors)],
        )

    date_str = date.strftime("%Y-%m-%d")
    fname = Path(mv_plot_dir, f"mv_oos_{end:04d}_{date_str}.png")

    mv_plot(mv_pairs=global_mv_pairs, save_path=str(fname), date=date,
            show_plot=show_plots, highlight_ptf=highlights, ef_pairs=eff_front)

# Batch Ploto Function
# --------------------


def do_plot_batch(end, date, mv_plot_dir, show_plots, mu_w: Union[np.ndarray, pd.Series],
                  cov_w: Union[np.ndarray, pd.DataFrame], n_ptfs: int, min_assets: int, max_assets: int,
                  random_state: int, window_mv_data) -> None:
    """
    Plots for batch and executes in parallel inside each thread
    """

    mv_pairs_t, _, _ = portfolio_sampler(mu_rets=mu_w, cov_rets=cov_w, n_portfolios=n_ptfs, min_assets=min_assets,
                                         max_assets=max_assets, random_state=random_state)

    eff_front, _ = compute_efficient_frontier(mu_w, cov_w, n_ptfs)

    plot_window_mv(end, date, mv_plot_dir, show_plots,
                   mv_pairs_t, eff_front, window_mv_data)

# Allocation Plot Function
# ------------------------


def plot_allocation_frame(weights_df: pd.DataFrame, end: int, date: pd.Timestamp, save_dir: str | Path,
                          show_plot: bool = False, asset_colors: Dict[str, Any] | None = None) -> None:
    """
    Plot of the allocation for each asset in a portfolio with a clean legend.
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
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    ax.set_title(f"Portfolio Allocations | {date_str}")
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

    fig.tight_layout()

    date_str = date.strftime("%Y-%m-%d")
    fname = save_dir / f"alloc_{end:04d}_{date_str}.png"
    fig.savefig(fname, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# Plot Cumulative OOS Function
# ----------------------------


def plot_cumulative_oos(oos_df: pd.DataFrame, save_path: str | Path, show_plot: bool = False, title: str = "Cumulative Out-of-Sample Performance (Excess Returns)") -> None:
    """
    Plot cumulative performance from returns.
    """

    oos_df = oos_df.copy().sort_index()
    cum = (1.0 + oos_df).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    cum.plot(ax=ax, linewidth=2)

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    image_path = Path(save_path, "cumulative_oos_ret.png")
    fig.savefig(image_path, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# Plot MV Grid Function
# ---------------------


def mv_plot_on_axis(ax: plt.Axes, cax: plt.Axes | None, mv_pairs: List[List[float]], date: pd.Timestamp, highlight_ptf: Dict[str, plot_ptf] | None = None,
                    ef_pairs: Union[np.ndarray, None] = None, show_colorbar: bool = True) -> None:
    """
    Same visual logic as mv_plot, but draws on a provided axis (for grids).
    """
    mv_pairs = np.asarray(mv_pairs, dtype=float)
    vols = np.sqrt(mv_pairs[:, 1])
    rets = mv_pairs[:, 0]

    sharpe = np.zeros_like(rets)
    nonzero = vols > 0
    sharpe[nonzero] = rets[nonzero] / vols[nonzero]

    sc = ax.scatter(
        vols, rets, c=sharpe, cmap="RdBu", s=30,
        edgecolors="black", linewidths=0.5
    )

    if show_colorbar and cax is not None:
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label("Sharpe (excess)", rotation=90)

    if ef_pairs is not None:
        ef_pairs = np.asarray(ef_pairs, dtype=float)
        ef_vol = np.sqrt(ef_pairs[:, 1])
        ef_ret = ef_pairs[:, 0]
        ax.plot(ef_vol, ef_ret, linewidth=2.5, label="Efficient Frontier")

    if highlight_ptf:
        for name, info in highlight_ptf.items():
            mv_pair = info.mv_pair
            color = info.color
            ret_h = float(mv_pair[0])
            vol_h = float(mv_pair[1]) ** 0.5

            ax.scatter([vol_h], [ret_h], marker="x", s=80,
                       color=color, label=name, zorder=5)
            ax.text(vol_h, ret_h, f" {name}", color=color,
                    fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Monthly Risk (Volatility)")
    ax.set_ylabel("Monthly Excess Return")
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    ax.set_title(f"Sample of Random Portfolios | {date_str}")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("white")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            loc="lower right", frameon=True, framealpha=0.7,
            facecolor="white", edgecolor="black", fontsize=9
        )
        legend.get_frame().set_linewidth(0.8)


def plot_mv_grid(img_dir: str | Path, date_strs: Sequence[str], out_name: str = "mv_grid_vertical.png", figsize: tuple[float, float] | None = None,
                 title: str | None = None, show_plot: bool = False) -> Path:
    """
    Build a vertical grid from already-generated images in img_dir, selecting images that match the provided date strings (YYYY-MM-DD).
    """
    img_dir = Path(img_dir)

    date_to_file: dict[str, Path] = {}
    all_imgs = sorted([p for p in img_dir.iterdir()
                      if p.suffix.lower() in [".png"]])

    for d in date_strs:
        match = next((p for p in all_imgs if f"_{d}" in p.stem), None)
        if match is not None:
            date_to_file[d] = match
        else:
            print(f"[warn] No image found for date {d} in {img_dir}")

    selected = [date_to_file[d] for d in date_strs if d in date_to_file]

    if len(selected) == 0:
        raise ValueError(
            f"No matching images found in {img_dir} for provided dates.")

    n_rows = len(selected)
    n_cols = 1

    if figsize is None:
        figsize = (10, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = [axes]

    for ax, img_path in zip(axes, selected):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    out_path = img_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# Plot Allocation Functions
# -------------------------


def plot_allocation_panel(ax, weights_df, asset_colors=None):
    """
    Plot a stacked bar chart of portfolio allocations on a given Axes object.
    """
    assets = list(weights_df.index)
    portfolios = list(weights_df.columns)
    n_ptf = len(portfolios)
    x = np.arange(1, n_ptf + 1)
    bottoms = np.zeros(n_ptf, dtype=float)
    weights_pct = weights_df.values * 100.0

    for i, asset in enumerate(assets):
        w = weights_pct[i, :]
        color = asset_colors.get(asset, None) if asset_colors else None
        ax.bar(x, w, bottom=bottoms, color=color, linewidth=0)
        bottoms += w

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_alloc_grid(alloc_snapshots: dict, save_path: str | Path, asset_colors: dict | None = None,
                    suptitle: str = "Portfolio Allocations Across Selected Dates") -> None:
    """
    Create and save a grid of portfolio allocation panels across selected dates.
    """
    dates = sorted(alloc_snapshots.keys())[:4]
    if len(dates) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, dt in zip(axes, dates):
        weights_df = alloc_snapshots[dt]
        plot_allocation_panel(ax, weights_df, asset_colors=asset_colors)
        ax.set_title(dt.strftime("%Y-%m-%d"))
        ax.set_ylabel("Allocation (%)")

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
