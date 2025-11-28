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
from utils.visualization import do_plot_batch

# Callable Function Structure
# ---------------------------

OptimFunc = Callable[
    [pd.Series, pd.DataFrame, int],
    Tuple[Dict[str, float], float, float]
]

# Build Universe Mask Function
# ----------------------------


def choose_avail_assets(train_full: pd.DataFrame, test_row_full: pd.Series, late_cols: list[str]) -> pd.Series:

    mask = test_row_full.notna().copy()

    for col in late_cols:
        if col in mask.index:
            full_history = train_full[col].notna().all()
            if not full_history or pd.isna(test_row_full[col]):
                mask[col] = False

    return mask

# Window Processing Function
# --------------------------


def process_window(returns: pd.DataFrame, end: int, window: int, min_assets: int) -> Union[tuple[pd.Timestamp, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, int], None]:
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


def oos_results_per_method(oos_results: Dict[str, list[float]], methods: Dict[str, Callable], mu_w: pd.Series, test_row: pd.Series,
                           cov_w: pd.DataFrame, n_obs: int) -> Dict[str, Tuple[float, float]]:
    """
    Compute Out-of-Sample results for the all the methods
    """

    window_mv_data: Dict[str, Tuple[float, float]] = {}

    for name, opt_fn in methods.items():

        weights_dict, r_m, vol_m = opt_fn(mu_w, cov_w, n_obs)

        w = pd.Series(weights_dict).reindex(cov_w.columns).fillna(0.0)
        r_oos = float(test_row @ w)

        oos_results[name].append(r_oos)

        window_mv_data[name] = (r_m, vol_m)

    return window_mv_data

# Print Window Results Function
# -----------------------------


def print_window_results(end: int, date: pd.Timestamp, rf_m: float, oos_results: Dict[str, List[float]], window_mv_data: Dict[str, Tuple[float, float]]) -> None:
    """
    Print window results (for each window of data).
    """
    print(f"\n----- Batch {end} | Date: {date} -----")

    for name, (r_m, vol_m) in window_mv_data.items():
        excess_m: float = r_m - rf_m
        last_oos: float = oos_results[name][-1]
        excess_oos: float = last_oos - rf_m

        print(f"[{name}] | Expected Return: {r_m:.4f} | Expected Vol: {vol_m:.4f} | Expected Excess: {excess_m:.4f} | OOS Return: {last_oos:.4f} | OOS Excess: {excess_oos: .4f}")

# Performance Statistics Function
# -------------------------------


def performance_stats(rf_ann: float, oos_df: pd.DataFrame, frequency: FreqPrices) -> pd.DataFrame:
    """
    Prints the performance stats of our results
    """
    rf_m: float = (1 + rf_ann)**(1/frequency.value) - 1
    stats: Dict[str, Dict[str, float]] = {}

    for method in oos_df.columns:
        r = oos_df[method].dropna()

        mean_m: float = r.mean()
        vol_m: float = r.std(ddof=1)

        excess_m: float = mean_m - rf_m
        sharpe_m: Union[float, None] = excess_m / \
            vol_m if vol_m > 0 else np.nan

        mean_ann: float = (1 + mean_m)**frequency.value - 1
        vol_ann: float = vol_m * (frequency.value**0.5)
        sharpe_ann: float = sharpe_m * (frequency.value**0.5)

        stats[method] = {"mean_monthly": mean_m, "vol_monthly": vol_m, "sharpe_monthly": sharpe_m, "ann_return": mean_ann, "ann_volatility": vol_ann,
                         "ann_sharpe": sharpe_ann}

    return pd.DataFrame(stats).T

# OOS Running Pipeline Function
# -----------------------------


def run_oos_backtest(returns: pd.DataFrame, frequency: FreqPrices, window: int, methods: Dict[str, OptimFunc], mv_plot_dir: str | Path, show_plots: bool = False,
                     do_plots: bool = False, print_res: bool = False, n_ptfs: int = 3000, min_assets: int = 1, max_assets: int | None = None, random_state: int = 123,
                     rf: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Out-of-Sample Backtest for the different methods and windows of data
    """

    rf_m: float = (1 + rf)**(1 / frequency.value) - 1

    oos_dates: List[str] = []
    oos_results: Dict[str, List[Any]] = {name: [] for name in methods.keys()}

    executor = ThreadPoolExecutor(max_workers=4)  # parallelization
    futures: List[Any] = []

    plot_pbar = None
    if do_plots:
        total_plots = len(returns) - window
        plot_pbar = tqdm(total=total_plots, desc="Saved Plots", unit="plot")

    for end in tqdm(range(window, len(returns)), desc="Simulated Batches", unit="batch"):

        processed: Union[Tuple[pd.Timestamp, pd.DataFrame, pd.Series, pd.Series,
                         pd.DataFrame, int], None] = process_window(returns, end, window, min_assets)

        if processed is None:
            if do_plots and plot_pbar:
                plot_pbar.update(1)
            continue

        date, _, test_row, mu_w, cov_w, n_obs = processed
        oos_dates.append(date)

        window_mv_data: Dict[str, Tuple[float, float]] = oos_results_per_method(
            oos_results, methods, mu_w, test_row, cov_w, n_obs)

        if print_res:
            print_window_results(end, date, rf_m, oos_results, window_mv_data)

        if do_plots:
            futures.append(executor.submit(do_plot_batch, end, date, mv_plot_dir, show_plots,
                           rf, mu_w, cov_w, n_ptfs, min_assets, max_assets, random_state, window_mv_data))

            plot_pbar.update(1)

    for f in futures:
        f.result()

    if do_plots and plot_pbar:
        plot_pbar.close()

    oos_df: pd.DataFrame = pd.DataFrame(oos_results, index=oos_dates)
    stats_df: pd.DataFrame = performance_stats(rf, oos_df, frequency)

    return oos_df, stats_df
