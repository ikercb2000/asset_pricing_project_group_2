# Packages
# --------

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple
from tqdm import tqdm
from IPython.display import Image, HTML, display
from pathlib import Path

# Project Modules
# ---------------

from utils.types import plot_ptf


# Data Analysis Plot Class
# ------------------------

class DataAnalysisPlots:
    def __init__(self, returns: pd.DataFrame, base_dir: str | Path):
        self.returns = returns
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _savefig(self, name: str):
        path = self.base_dir / f"{name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_boxplots(self):
        """
        Plot boxplot plot for the data.
        """
        plt.figure(figsize=(10, 5))
        self.returns.boxplot(rot=90)
        plt.title("Boxplots of Monthly Returns")
        self._savefig("boxplots")

    def plot_corr_heatmap(self):
        """
        Plot correlation heatmap matrix.
        """
        plt.figure(figsize=(10, 5))
        sns.heatmap(self.returns.corr(), cmap="RdBu", center=0)
        plt.title("Correlation Heatmap")
        self._savefig("correlation_heatmap")

    def plot_rolling_vol(self, window: int = 24, legend_mode: str = "horizontal"):
        """
        Plot rolling volatility with improved layout.
        """
        rolling_vol = self.returns.rolling(window).std()

        if legend_mode == "subset":
            avg_vol = rolling_vol.mean().sort_values(ascending=False)
            selected = avg_vol.index[:5]
            rolling_to_plot = rolling_vol[selected]
        else:
            rolling_to_plot = rolling_vol

        plt.figure(figsize=(4, 8))

        ax = rolling_to_plot.plot(alpha=0.9, linewidth=1.2)
        plt.title(f"Rolling Volatility ({window}-month)")
        plt.xlabel("Date")
        plt.ylabel("Volatility")

        # LEGEND HANDLING
        if legend_mode == "horizontal":
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=5,
                fontsize=9,
                frameon=False
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

    def print_cov_condition(self):
        """
        Print condition number of covariance matrix.
        """
        cov = self.returns.cov()
        cond = np.linalg.cond(cov)
        print("--- Covariance Matrix Condition Number ---")
        print(f"Condition number: {cond: .3e}")
        eigvals = np.linalg.eigvals(cov)
        print(f"Min eigenvalue : {eigvals.min(): .3e}")
        print("------------------------------------------")

    def print_describe(self):
        """
        Print descriptive table for analysing data.
        """
        desc = self.returns.describe().T
        desc["n_missing"] = self.returns.isna().sum()
        desc["var"] = self.returns.var()
        cols = ["count", "n_missing", "mean", "std",
                "var", "min", "25%", "50%", "75%", "max"]
        desc = desc[cols]
        print("--- Descriptive statistics of monthly returns ---")
        display(desc)

    # 6) Mostrar plots uno debajo de otro con espacio
    def display_plots(self, names: list[str]):
        """
        Shows the plots computed before.
        """
        for i, name in enumerate(names):
            img_path = self.base_dir / f"{name}.png"
            if not img_path.exists():
                print(f"Plot file not found: {img_path}")
                continue
            display(Image(filename=str(img_path)))
            if i < len(names) - 1:
                display(HTML("<br><br>"))

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


def portfolio_sampler(
    mu_rets: pd.Series,
    cov_rets: pd.DataFrame,
    n_portfolios: int,
    min_assets: int = 1,
    max_assets: int | None = None,
    random_state: int | None = None,
    net_exposure: float = 1.0,
    gross_limit: float = 3.0,
) -> Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:

    np.random.seed(random_state)

    stocks_list = list(cov_rets.columns)

    if max_assets is None:
        max_assets = len(stocks_list)

    mean_var_pairs = []
    weights_list = []
    tickers_list = []

    pbar = tqdm(total=n_portfolios, desc="Sampling portfolios")

    while len(mean_var_pairs) < n_portfolios:

        k = np.random.randint(min_assets, max_assets + 1)
        selected_assets = np.random.choice(stocks_list, k, replace=False)

        w = random_w_leverage(
            k,
            net_exposure=net_exposure,
            gross_limit=gross_limit,
        )

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

def mv_plot(mv_pairs: List[List[float]], save_path: str, rf: float = 0.05, highlight_ptf: plot_ptf = None) -> None:
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

    fig.write_image(save_path, scale=2)
    display(Image(filename=save_path))
