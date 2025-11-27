# Packages
# --------

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from typing import List, Tuple, Dict
from tqdm import tqdm
from IPython.display import Image, HTML, display
from pathlib import Path

# Project Modules
# ---------------

from utils.types import plot_ptf
from utils.optimisers import compute_efficient_frontier

# Data Analysis Plot Class
# ------------------------


class DataAnalysisPlots:
    def __init__(self, returns: pd.DataFrame, base_dir: str | Path):
        self.returns = returns
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _savefig(self, name: str):
        """
        Save figures as if manually done in a Jupyter notebook.
        """
        path = Path(self.base_dir, f"{name}.png")
        plt.tight_layout()             # típico en notebooks
        # sin bbox_inches="tight"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        # NOTA: no cerramos la figura → comportamiento notebook

    def plot_boxplots(self):
        plt.figure(figsize=(14, 4))
        self.returns.boxplot(rot=90)
        plt.title("Boxplots of Monthly Returns")
        self._savefig("boxplots")
        plt.show()                     # comportamiento normal notebook

    def plot_corr_heatmap(self):
        plt.figure(figsize=(14, 6))
        sns.heatmap(self.returns.corr(), cmap="RdBu", center=0)
        plt.title("Correlation Heatmap")
        self._savefig("correlation_heatmap")
        plt.show()

    def plot_rolling_vol(self, window: int = 24, legend_mode: str = "horizontal"):
        rolling_vol = self.returns.rolling(window).std()

        if legend_mode == "subset":
            avg_vol = rolling_vol.mean().sort_values(ascending=False)
            selected = avg_vol.index[:5]
            rolling_to_plot = rolling_vol[selected]
        else:
            rolling_to_plot = rolling_vol

        plt.figure(figsize=(24, 10))

        ax = rolling_to_plot.plot(alpha=0.9, linewidth=1)
        plt.title(f"Rolling Volatility ({window}-month)")
        plt.xlabel("Date")
        plt.ylabel("Volatility")

        if legend_mode == "horizontal":
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=5,
                fontsize=9,
                frameon=False,
            )
        elif legend_mode == "subset":
            plt.legend(
                loc='upper right',
                fontsize=9,
                frameon=False,
                title="Selected Assets"
            )
        elif legend_mode is None:
            ax.get_legend().remove()

        self._savefig("rolling_vol")
        plt.show()

    def print_cov_condition(self):
        cov = self.returns.cov()
        cond = np.linalg.cond(cov)
        print("--- Covariance Matrix Condition Number ---")
        print(f"Condition number: {cond: .3e}")
        eigvals = np.linalg.eigvals(cov)
        print(f"Min eigenvalue : {eigvals.min(): .3e}")
        print("------------------------------------------")

    def print_describe(self):
        desc = self.returns.describe().T
        desc["n_missing"] = self.returns.isna().sum()
        desc["var"] = self.returns.var()
        cols = ["count", "n_missing", "mean", "std",
                "var", "min", "25%", "50%", "75%", "max"]
        desc = desc[cols]
        print("\n--- Descriptive statistics of monthly returns ---")
        display(desc)

# Random Weights Function
# -----------------------


def random_w_leverage(
    k: int,
    net_exposure: float = 1.0,
    gross_limit: float = 2.0,
) -> np.ndarray:
    """
    Obtains random weights that allow leverage and short-selling
    """

    w = np.random.randn(k)
    w = w / np.sum(w) * net_exposure

    gross = np.sum(np.abs(w))
    if gross > gross_limit:
        w = w / gross * gross_limit

    return w

# Portfolio Sampler Function
# --------------------------


def _sample_one(
    mu_arr, cov_arr, tickers, n_assets_total,
    min_assets, max_assets, net_exposure, gross_limit, rng_seed
):
    rng = np.random.default_rng(rng_seed)

    k = rng.integers(min_assets, max_assets + 1)
    idx = rng.choice(n_assets_total, size=k, replace=False)

    w = random_w_leverage(k, net_exposure=net_exposure,
                          gross_limit=gross_limit)

    mu_sel = mu_arr[idx]
    cov_sel = cov_arr[np.ix_(idx, idx)]

    ret = float(w @ mu_sel)
    var = float(w @ cov_sel @ w)

    return (ret, var, w, tickers[idx])


def portfolio_sampler(mu_rets, cov_rets, n_portfolios, min_assets=1, max_assets=None, random_state=123, n_jobs=-1, net_exposure=1.0, gross_limit=3.0):
    tickers = np.array(cov_rets.columns)
    mu_arr = mu_rets.values
    cov_arr = cov_rets.values
    n_assets_total = len(tickers)

    if max_assets is None:
        max_assets = n_assets_total

    seeds = np.random.SeedSequence(random_state).spawn(n_portfolios)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_sample_one)(
            mu_arr, cov_arr, tickers,
            n_assets_total,
            min_assets, max_assets,
            net_exposure, gross_limit,
            s.generate_state(1)[0]  # each worker gets its own seed
        )
        for s in seeds
    )

    mean_var_pairs = [(r, v) for (r, v, _, _) in results]
    weights_list = [w for (_, _, w, _) in results]
    tickers_list = [t for (_, _, _, t) in results]

    return mean_var_pairs, weights_list, tickers_list


# Mean-Variance Plot
# ------------------

def mv_plot(
    mv_pairs: List[List[float]],
    save_path: str,
    show_plot: bool = True,
    rf: float = 0.05,
    highlight_ptf: plot_ptf = None,
    ef_pairs: np.ndarray | None = None,   # NEW
) -> None:
    """
    Plot Mean-Variance Frontier Graph with optional highlight portfolios
    and an optional efficient frontier line.
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
    if ef_pairs is not None:
        ef_pairs = np.asarray(ef_pairs)
        ef_vol = np.sqrt(ef_pairs[:, 1])
        ef_ret = ef_pairs[:, 0]

        fig.add_trace(
            go.Scatter(
                x=ef_vol,
                y=ef_ret,
                mode="lines",
                line=dict(width=3),
                name="Efficient Frontier",
            )
        )

    if highlight_ptf:
        for name, info in highlight_ptf.items():
            mv_pair = info.mv_pair
            color = info.color
            ret = float(mv_pair[0])
            vol = float(mv_pair[1])**0.5

            fig.add_trace(
                go.Scatter(
                    x=[vol], y=[ret],
                    mode="markers+text",
                    marker=dict(color=color, size=12, symbol="x"),
                    text=[name], textposition="top center",
                    name=name
                )
            )

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Monthly Risk (Volatility)'),
        yaxis=dict(title='Monthly Return'),
        title='Sample of Random Portfolios',
        width=850,
        height=500,
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="trace",
        ),
    )

    fig.write_image(save_path, scale=2)
    if show_plot:
        display(Image(filename=save_path))

# Plot Window MV Function
# -----------------------


def plot_window_mv(end: int, date: pd.Timestamp, mv_plot_dir: str | Path, show_plots: bool, rf: float, global_mv_pairs, eff_front: np.ndarray,
                   window_mv_data: Dict[str, Tuple[float, float]]) -> None:

    highlights: Dict[str, plot_ptf] = {}
    colors = ["green", "orange", "purple", "black", "red", "blue"]

    for i, (m_name, (r_m, vol_m)) in enumerate(window_mv_data.items()):
        highlights[m_name] = plot_ptf(
            mv_pair=(r_m, vol_m**2),   # mensual
            color=colors[i % len(colors)],
        )

    date_str = str(date).replace(" ", "-")
    fname = Path(mv_plot_dir, f"mv_oos_{end:04d}_{date_str}.png")

    mv_plot(
        mv_pairs=global_mv_pairs,
        save_path=str(fname),
        show_plot=show_plots,
        rf=rf,
        highlight_ptf=highlights,
        ef_pairs=eff_front)

# Batch Ploto Function
# --------------------


def _do_plot_batch(end, date, mv_plot_dir: str, show_plots, rf: float, mu_w: np.ndarray | pd.Series, cov_w, n_ptfs, min_assets, max_assets,
                   random_state, window_mv_data) -> None:
    """
    Plots for batch and executes in parallel inside each thread
    """
    mv_pairs_t, _, _ = portfolio_sampler(
        mu_rets=mu_w,
        cov_rets=cov_w,
        n_portfolios=n_ptfs,
        min_assets=min_assets,
        max_assets=max_assets,
        random_state=random_state,
    )

    eff_front = compute_efficient_frontier(mu_w, cov_w, n_ptfs)

    plot_window_mv(
        end, date, mv_plot_dir, show_plots, rf,
        mv_pairs_t, eff_front, window_mv_data
    )
