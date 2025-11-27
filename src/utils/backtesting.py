# Packages
# --------

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from typing import Callable, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

# Project Modules
# ---------------

from utils.types import plot_ptf
from utils.enums import FreqPrices
from utils.helpers import ann_rets, cov_mat
from utils.visualization import portfolio_sampler, _do_plot_batch

# Callable Structure
# ------------------

OptimFunc = Callable[
    [pd.Series, pd.DataFrame, int],
    Tuple[Dict[str, float], float, float]
]

# OOS Results per Method Function
# -------------------------------


def oos_results_per_method(oos_results: Dict[str, list[float]], methods: Dict[str, Callable], mu_w: pd.Series, test_row: pd.Series,
                           cov_w: pd.DataFrame, n_obs: int) -> Dict[str, Tuple[float, float]]:

    window_mv_data: Dict[str, Tuple[float, float]] = {}

    for name, opt_fn in methods.items():

        weights_dict, r_m, vol_m = opt_fn(mu_w, cov_w, n_obs)

        w = pd.Series(weights_dict).reindex(cov_w.columns).fillna(0.0)
        r_oos = float(test_row @ w)

        oos_results[name].append(r_oos)

        window_mv_data[name] = (r_m, vol_m)

    return window_mv_data

# Performance Statistics Function
# -------------------------------


def performance_stats(rf_ann: float, oos_df: pd.DataFrame, frequency: FreqPrices) -> pd.DataFrame:

    periods = frequency.value     # 12 para mensual
    rf_m = (1 + rf_ann)**(1/periods) - 1   # rf mensual

    stats = {}

    for method in oos_df.columns:
        r = oos_df[method].dropna()

        mean_m = r.mean()
        vol_m = r.std(ddof=1)

        excess_m = mean_m - rf_m
        sharpe_m = excess_m / vol_m if vol_m > 0 else np.nan

        mean_ann = (1 + mean_m)**periods - 1
        vol_ann = vol_m * (periods**0.5)
        sharpe_ann = sharpe_m * (periods**0.5)

        stats[method] = {
            "mean_monthly": mean_m,
            "vol_monthly": vol_m,
            "sharpe_monthly": sharpe_m,
            "ann_return": mean_ann,
            "ann_volatility": vol_ann,
            "ann_sharpe": sharpe_ann,
        }

    return pd.DataFrame(stats).T

# Global MV Pairs for Plot Function
# ---------------------------------


def build_global_mv_pairs(returns: pd.DataFrame, n_ptfs: int, min_assets: int, max_assets: int | None, random_state: int) -> list[list[float]]:

    mu_init = returns.mean()
    cov_init = returns.cov()

    global_mv_pairs, _, _ = portfolio_sampler(mu_rets=mu_init, cov_rets=cov_init, n_portfolios=n_ptfs, min_assets=min_assets,
                                              max_assets=max_assets, random_state=random_state)
    return global_mv_pairs

# Build Universe Mask Function
# ----------------------------


def build_universe_mask(train_full: pd.DataFrame, test_row_full: pd.Series, late_cols: list[str]) -> pd.Series:

    mask = test_row_full.notna().copy()

    for col in late_cols:
        if col in mask.index:
            full_history = train_full[col].notna().all()
            if not full_history or pd.isna(test_row_full[col]):
                mask[col] = False

    return mask

# Window Processing Function
# --------------------------


def process_window(returns: pd.DataFrame, end: int, window: int, late_cols: list[str],
                   min_assets: int, frequency: FreqPrices) -> tuple[pd.Timestamp, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, int] | None:

    train_full = returns.iloc[end - window:end]
    test_row_full = returns.iloc[end]
    date = returns.index[end]

    mask = build_universe_mask(train_full, test_row_full, late_cols)
    train = train_full.loc[:, mask]
    test_row = test_row_full[mask]

    if train.shape[1] < min_assets:
        return None

    mu_w = train.mean()
    cov_w = train.cov()
    n_obs = len(train)

    return date, train, test_row, mu_w, cov_w, n_obs


# OOS Running Pipeline Function
# -----------------------------


def run_oos_backtest(
        returns: pd.DataFrame,
        frequency: FreqPrices,
        window: int,
        methods: Dict[str, OptimFunc],
        mv_plot_dir: str | Path,
        show_plots: bool = False,
        do_plots: bool = False,
        print_res: bool = False,
        n_ptfs: int = 3000,
        min_assets: int = 1,
        max_assets: int | None = None,
        random_state: int = 123,
        rf: float = 0.0):

    rf_m = (1 + rf)**(1 / frequency.value) - 1

    late_cols = ["DOW", "V"]

    oos_dates = []
    oos_results = {name: [] for name in methods.keys()}

    # === THREAD EXECUTOR PARA PARALELIZAR PLOTS + SAMPLER ===
    executor = ThreadPoolExecutor(max_workers=4)
    futures = []

    plot_pbar = None
    if do_plots:
        total_plots = len(returns) - window
        plot_pbar = tqdm(total=total_plots, desc="Saved Plots", unit="plot")

    for end in tqdm(range(window, len(returns)), desc="Simulated Batches", unit="batch"):

        processed = process_window(
            returns, end, window, late_cols, min_assets, frequency
        )

        if processed is None:
            if do_plots and plot_pbar:
                plot_pbar.update(1)
            continue

        date, _, test_row, mu_w, cov_w, n_obs = processed
        oos_dates.append(date)

        window_mv_data = oos_results_per_method(
            oos_results, methods, mu_w, test_row, cov_w, n_obs
        )

        if print_res:
            print(f"\n===== Batch {end} | Date: {date} =====")
            for name, (r_m, vol_m) in window_mv_data.items():
                excess_m = r_m - rf_m
                last_oos = oos_results[name][-1]
                excess_oos = last_oos - rf_m

                print(
                    f"[{name}] "
                    f"Expected Return: {r_m:.6f}, "
                    f"Expected Vol: {vol_m:.6f}, "
                    f"Expected Excess: {excess_m:.6f}, "
                    f"OOS Return: {last_oos:.6f}, "
                    f"OOS Excess: {excess_oos:.6f}"
                )

        if do_plots:

            # ⚡ Enviar sampler + plot a un thread paralelo
            futures.append(
                executor.submit(
                    _do_plot_batch,
                    end, date, mv_plot_dir, show_plots, rf,
                    mu_w, cov_w, n_ptfs, min_assets, max_assets,
                    random_state, window_mv_data
                )
            )

            plot_pbar.update(1)

    for f in futures:
        f.result()

    if do_plots and plot_pbar:
        plot_pbar.close()

    oos_df = pd.DataFrame(oos_results, index=oos_dates)
    stats_df = performance_stats(rf, oos_df, frequency)

    return oos_df, stats_df
