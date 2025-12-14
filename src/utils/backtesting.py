# Packages
# --------

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from typing import Callable, Tuple, Dict, Union, List, Any
from pathlib import Path
from tqdm import tqdm

# Project Modules
# ---------------

from utils.enums import FreqPrices
from utils.visualization import do_plot_batch, plot_allocation_frame, plot_cumulative_oos, plot_mv_grid, plot_alloc_grid
from utils.helpers import generate_asset_colors  # <- colores globales

OptimFunc = Callable[
    [pd.Series, pd.DataFrame, int],
    Tuple[Dict[str, float], float, float]
]


# Build Universe Mask Function
# ----------------------------

def choose_avail_assets(train_full: pd.DataFrame, test_row_full: pd.Series,
                        late_cols: list[str]) -> pd.Series:

    mask = test_row_full.notna().copy()

    for col in late_cols:
        if col in mask.index:
            full_history = train_full[col].notna().all()
            if not full_history or pd.isna(test_row_full[col]):
                mask[col] = False

    return mask


# Window Processing Function
# --------------------------

def process_window(returns: pd.DataFrame, end: int, window: int,
                   min_assets: int) -> Union[
                       tuple[pd.Timestamp, pd.DataFrame, pd.Series,
                             pd.Series, pd.DataFrame, int],
                       None]:
    """
    For each window, process the data to obtain necessary train, test and other matrices and vectors
    """

    late_cols = ["DOW", "V"]

    train_full: pd.DataFrame = returns.iloc[end - window:end]
    test_row_full: pd.Series = returns.iloc[end]
    date = returns.index[end]

    # Choose only available assets
    mask = choose_avail_assets(train_full, test_row_full, late_cols)
    train: pd.DataFrame = train_full.loc[:, mask]
    test_row: pd.Series = test_row_full[mask]

    if train.shape[1] < min_assets:
        return None

    mu_w: pd.Series = train.mean()
    cov_w: pd.DataFrame = train.cov()
    n_obs: int = len(train)

    return date, train, test_row, mu_w, cov_w, n_obs


# OOS Results per Method Function
# -------------------------------

def oos_results_per_method(
        oos_results: Dict[str, list[float]], methods: Dict[str, Callable], mu_w: pd.Series, test_row: pd.Series,
        cov_w: pd.DataFrame, n_obs: int) -> Dict[str, Tuple[float, float]]:
    """
    Compute Out-of-Sample results for the all the methods
    """

    window_mv_data: Dict[str, Tuple[float, float]] = {}
    window_weights: Dict[str, pd.Series] = {}

    for name, opt_fn in methods.items():
        weights_dict, r_m, vol_m = opt_fn(mu_w, cov_w, n_obs)
        w = pd.Series(weights_dict).reindex(cov_w.columns).fillna(0.0)
        r_oos = float(test_row @ w)
        oos_results[name].append(r_oos)
        window_mv_data[name] = (r_m, vol_m)
        window_weights[name] = w

    return window_mv_data, window_weights


# Print Window Results Function
# -----------------------------

def print_window_results(end: int, date: pd.Timestamp, oos_results: Dict[str, List[float]], window_mv_data: Dict[str, Tuple[float, float]]) -> None:
    """
    Print window results (for each window of data).
    """
    print(f"\n----- Batch {end} | Date: {date} -----")

    for name, (r_m, vol_m) in window_mv_data.items():
        last_oos: float = oos_results[name][-1]
        print(
            f"[{name}] | Expected Return: {r_m:.4f} | Expected Vol: {vol_m:.4f} "
            f"| Expected Excess: {r_m:.4f} | OOS Excess Return: {last_oos:.4f}"
        )


# Performance Statistics Function
# -------------------------------

def performance_stats(oos_df: pd.DataFrame, frequency: FreqPrices = FreqPrices.MONTHLY, weights_by_method: Dict[str, pd.DataFrame] | None = None, benchmark: str | None = None,
                      var_alpha: float = 0.05, sortino_target: float = 0.0, rolling_sharpe_window: int = 24) -> pd.DataFrame:
    """
    Prints the performance stats of our results
    """
    stats: Dict[str, Dict[str, float]] = {}

    for method in oos_df.columns:
        r = oos_df[method].dropna()
        excess_m: float = r.mean()
        vol_m: float = r.std(ddof=1)
        sharpe_m: float = excess_m / vol_m if vol_m > 0 else np.nan

        downside = np.minimum(r - sortino_target, 0.0)
        downside_dev = np.sqrt(np.mean(downside**2))
        sortino = excess_m / downside_dev if downside_dev > 0 else np.nan

        skew = r.skew()
        kurt = r.kurt()

        var_q = np.quantile(r.values, var_alpha)
        var = -var_q

        roll_mean = r.rolling(rolling_sharpe_window).mean()
        roll_std = r.rolling(rolling_sharpe_window).std(ddof=1)
        roll_sharpe = roll_mean / roll_std
        sharpe_std = roll_sharpe.dropna().std(ddof=1)

        info_ratio = np.nan
        if benchmark is not None and benchmark in oos_df.columns and benchmark != method:
            rb = oos_df[benchmark].reindex(r.index).dropna()
            rr = r.reindex(rb.index)
            active = rr - rb
            te = active.std(ddof=1)
            info_ratio = active.mean() / te if te > 0 else np.nan

        turnover = np.nan
        hhi = np.nan
        if weights_by_method is not None and method in weights_by_method:
            W = weights_by_method[method].reindex(r.index).fillna(0.0)
            turnover = W.diff().abs().sum(axis=1).dropna().mean()
            hhi = (W.pow(2).sum(axis=1)).mean()

        stats[method] = {
            "mean_monthly": excess_m,
            "vol_monthly": vol_m,
            "sharpe_monthly": sharpe_m,
            "sortino": sortino,
            "skewness": skew,
            "kurtosis_excess": kurt,
            f"VaR_{100-int(var_alpha*100)}%": var,
            "info_ratio": info_ratio,
            "turnover": turnover,
            "weight_concentration_HHI": hhi,
            "std_rolling_sharpe": sharpe_std,
        }

    return pd.DataFrame(stats).T

# OOS Running Pipeline Function
# -----------------------------


def run_oos_backtest(
    returns: pd.DataFrame,
    window: int,
    methods: Dict[str, OptimFunc],
    mv_plot_dir: str | Path,
    frequency: FreqPrices = FreqPrices.MONTHLY,
    show_plots: bool = False,
    grid_dates: list[str] | None = None,
    print_res: bool = False,
    n_ptfs: int = 3000,
    min_assets: int = 1,
    max_assets=None,
    random_state: int = 123,
    alloc_plot_dir: str | Path | None = None,
    oos_ret_cum_dir: str | Path | None = None,
    grid_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run Out-of-Sample Backtest for the different methods and windows of data and create plots.
    """

    mv_plot_dir = Path(mv_plot_dir)
    if alloc_plot_dir is not None:
        alloc_plot_dir = Path(alloc_plot_dir)

    if grid_dir is not None:
        grid_dir = Path(grid_dir)
        grid_dir.mkdir(parents=True, exist_ok=True)

    oos_dates: List[pd.Timestamp] = []
    oos_results: Dict[str, List[Any]] = {name: [] for name in methods.keys()}
    weights_hist: Dict[str, List[pd.Series]] = {
        name: [] for name in methods.keys()}

    executor = ThreadPoolExecutor(max_workers=4)
    futures: List[Any] = []

    asset_colors = None
    if alloc_plot_dir is not None:
        all_assets = list(returns.columns)
        asset_colors = generate_asset_colors(all_assets)

    for end in tqdm(range(window, len(returns)), desc="Simulated Batches", unit="batch"):

        processed = process_window(returns, end, window, min_assets)
        if processed is None:
            continue

        date, _, test_row, mu_w, cov_w, n_obs = processed
        oos_dates.append(date)

        # compute OOS results
        window_mv_data, window_weights = oos_results_per_method(
            oos_results=oos_results,
            methods=methods,
            mu_w=mu_w,
            test_row=test_row,
            cov_w=cov_w,
            n_obs=n_obs
        )

        for name, w in window_weights.items():
            weights_hist[name].append(w)

        if print_res:
            print_window_results(end, date, oos_results, window_mv_data)

        # MV plot saved as image (already)
        futures.append(
            executor.submit(
                do_plot_batch,
                end, date, mv_plot_dir, show_plots,
                mu_w, cov_w, n_ptfs, min_assets, max_assets,
                random_state, window_mv_data
            )
        )

        # Allocation plot saved as image (already)
        if alloc_plot_dir is not None:
            weights_df = pd.DataFrame(window_weights)
            plot_allocation_frame(
                weights_df=weights_df,
                end=end,
                date=date,
                save_dir=alloc_plot_dir,
                show_plot=False,
                asset_colors=asset_colors
            )

    # ensure MV plots are saved
    for f in tqdm(futures, desc="Saved MV Plots"):
        f.result()

    # build OOS returns df
    oos_df: pd.DataFrame = pd.DataFrame(oos_results, index=oos_dates)

    # cumulative OOS plot
    if oos_ret_cum_dir is not None:
        plot_cumulative_oos(
            oos_df=oos_df,
            save_path=oos_ret_cum_dir,
            show_plot=show_plots,
            title="Cumulative Out-of-Sample Performance (Excess Returns)",
        )

    # weights history by method
    weights_by_method = {
        name: pd.DataFrame(weights_hist[name], index=oos_dates).fillna(0.0)
        for name in weights_hist
    }

    stats_df = performance_stats(
        oos_df,
        frequency=frequency,
        weights_by_method=weights_by_method,
        benchmark="1/N-Ptf"
    )

    # -----------------------------
    # FINAL: Build vertical grids
    # -----------------------------
    if grid_dates:
        # MV grid
        mv_out = (mv_plot_dir / "mv_grid_vertical.png")
        tmp_mv = plot_mv_grid(
            img_dir=mv_plot_dir,
            date_strs=grid_dates,
            out_name=mv_out.name,
            title="Mean-Variance Plots | Selected Dates",
            show_plot=False,
        )
        # If grid_dir != mv_plot_dir, move file to grid_dir
        if grid_dir and mv_plot_dir != grid_dir:
            (grid_dir / tmp_mv.name).write_bytes(tmp_mv.read_bytes())
            tmp_mv.unlink()

        # Allocations grid
        if alloc_plot_dir is not None:
            alloc_out = (grid_dir / "alloc_grid_vertical.png")
            tmp_alloc = plot_mv_grid(
                img_dir=alloc_plot_dir,
                date_strs=grid_dates,
                out_name=alloc_out.name,
                title="Allocations | Selected Dates",
                show_plot=False,
            )
            if grid_dir and alloc_plot_dir != grid_dir:
                (grid_dir / tmp_alloc.name).write_bytes(tmp_alloc.read_bytes())
                tmp_alloc.unlink()

    return oos_df, stats_df, weights_by_method
