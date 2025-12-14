# Packages
# --------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.dates as mdates

from matplotlib.ticker import MaxNLocator
from pathlib import Path
from IPython.display import display
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


class DataAnalysisPlots:
    def __init__(self, returns: pd.DataFrame, base_dir: str | Path):
        self.returns = returns
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --------------- helpers -----------------

    def _savefig(self, name: str) -> None:
        """
        Save figure directly inside self.base_dir (no subfolders).
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")

    def _format_date_axis(self, ax, n_labels: int = 8) -> None:
        """
        Formats temporal axis.
        """
        ax.xaxis.set_major_locator(MaxNLocator(n_labels))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _asset_tag(self) -> str:
        """
        If one column -> return asset ticker, else 'ALL'.
        """
        return self.returns.columns[0] if self.returns.shape[1] == 1 else "ALL"

    # --------------- universe-level plots -----------------

    def plot_boxplots_all_assets(self) -> None:
        """
        Boxplots for the whole universe (saved in base_dir).
        """
        plt.figure(figsize=(14, 4))
        self.returns.boxplot(rot=90)
        plt.title("Boxplots of Monthly Returns — All Assets")
        self._savefig("boxplots_all_assets")
        plt.close()

    def plot_corr_heatmap(self) -> None:
        """
        Correlation matrix for the whole universe.
        This is the ONLY plot shown on screen.
        """
        plt.figure(figsize=(14, 6))
        sns.heatmap(self.returns.corr(), cmap="RdBu", center=0)
        plt.title("Correlation Heatmap — All Assets")
        self._savefig("correlation_heatmap_all_assets")
        plt.show()
        plt.close()

    def plot_rolling_corr_all_assets(self, window: int = 36) -> None:
        """
        Average pairwise rolling correlation of the universe (saved in base_dir).
        """
        rolling_corr = self.returns.rolling(window).corr()
        avg_corr = rolling_corr.groupby(level=0).mean()

        plt.figure(figsize=(14, 5))
        ax = plt.gca()
        ax.plot(avg_corr.mean(axis=1))

        ax.set_title(
            f"Average Rolling Correlation ({window}-month window) — All Assets")
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.grid(True, linestyle="--", alpha=0.3)

        self._format_date_axis(ax)
        self._savefig("rolling_avg_correlation_all_assets")
        plt.close()

    def plot_pca_variance(self) -> None:
        """
        PCA on the whole panel (saved in base_dir).
        """
        if self.returns.shape[1] < 2:
            print("PCA variance plot skipped: only one asset.")
            return

        X = self.returns.dropna().values
        pca = PCA().fit(X)

        plt.figure(figsize=(12, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.title("Cumulative Variance Explained by PCA — All Assets")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.grid(True, linestyle="--", alpha=0.3)

        self._savefig("pca_cumulative_variance_all_assets")
        plt.close()

    # --------------- per-asset plots -----------------

    def plot_boxplots_asset(self) -> None:
        tag = self._asset_tag()
        plt.figure(figsize=(10, 4))
        self.returns.boxplot(rot=0)
        plt.title(f"Boxplots of Monthly Returns — {tag}")
        self._savefig(f"boxplots_{tag}")
        plt.close()

    def plot_return_distribution_asset(self, bins: int = 30) -> None:
        tag = self._asset_tag()
        series = self.returns.iloc[:, 0].dropna()

        plt.figure(figsize=(10, 4))
        sns.histplot(series, kde=True, bins=bins)
        plt.title(f"Return Distribution — {tag}")

        self._savefig(f"return_distribution_{tag}")
        plt.close()

    def plot_rolling_vol_asset(self, window: int = 24) -> None:
        tag = self._asset_tag()
        series = self.returns.iloc[:, 0]
        rolling_vol = series.rolling(window).std()

        plt.figure(figsize=(14, 5))
        ax = plt.gca()

        ax.fill_between(rolling_vol.index, 0, rolling_vol, alpha=0.3)
        ax.plot(rolling_vol.index, rolling_vol, linewidth=2)

        ax.set_title(f"Rolling Volatility ({window}-month window) — {tag}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.grid(True, linestyle="--", alpha=0.4)

        self._format_date_axis(ax)
        self._savefig(f"rolling_volatility_{tag}")
        plt.close()

    def plot_rolling_sharpe_asset(self, window: int = 24) -> None:
        tag = self._asset_tag()
        series = self.returns.iloc[:, 0]

        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std

        plt.figure(figsize=(14, 5))
        ax = plt.gca()
        ax.plot(rolling_sharpe)

        ax.set_title(f"Rolling Sharpe Ratio ({window}-month window) — {tag}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe")
        ax.grid(True, linestyle="--", alpha=0.3)

        self._format_date_axis(ax)
        self._savefig(f"rolling_sharpe_{tag}")
        plt.close()

    def plot_drawdowns_asset(self) -> None:
        tag = self._asset_tag()
        series = self.returns.iloc[:, 0]

        cumulative = (1 + series).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1

        plt.figure(figsize=(14, 5))
        ax = plt.gca()
        ax.plot(drawdown)

        ax.set_title(f"Drawdowns — {tag}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.grid(True, linestyle="--", alpha=0.4)

        self._format_date_axis(ax)
        self._savefig(f"drawdowns_{tag}")
        plt.close()

    def plot_qq_asset(self) -> None:
        tag = self._asset_tag()
        series = self.returns.iloc[:, 0].dropna()

        plt.figure(figsize=(6, 6))
        stats.probplot(series, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot — {tag}")

        self._savefig(f"qq_plot_{tag}")
        plt.close()

    def plot_grid_for_tickers(self, tickers: list[str], n_rows: int, n_cols: int, save_prefix: str, window: int = 24, bins: int = 30) -> None:

        plot_kinds = ["drawdowns", "volatility", "sharpe", "qq", "hist"]

        for kind in plot_kinds:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows)
            )
            axes = axes.flatten()

            for ax, tkr in zip(axes, tickers[: n_rows * n_cols]):
                if tkr not in self.returns.columns:
                    ax.axis("off")
                    continue

                s = self.returns[tkr].dropna()
                if s.empty:
                    ax.axis("off")
                    continue

                if kind == "drawdowns":
                    cum = (1 + s).cumprod()
                    dd = cum / cum.cummax() - 1
                    ax.plot(dd)

                elif kind == "volatility":
                    rv = s.rolling(window).std()
                    ax.plot(rv)

                elif kind == "sharpe":
                    m = s.rolling(window).mean()
                    sd = s.rolling(window).std()
                    ax.plot(m / sd)

                elif kind == "qq":
                    stats.probplot(s, dist="norm", plot=ax)

                elif kind == "hist":
                    sns.histplot(
                        s,
                        bins=bins,
                        kde=True,
                        stat="density",
                        ax=ax,
                    )

                ax.set_title(tkr)
                ax.grid(True, linestyle="--", alpha=0.3)

            # Hide unused axes
            for ax in axes[len(tickers[: n_rows * n_cols]):]:
                ax.axis("off")

            fig.suptitle(f"{kind.upper()} | Selected Assets", fontsize=14)
            fig.tight_layout()

            path = self.base_dir / f"{save_prefix}_{kind}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --------------- summary tables (all assets at once) -----------------

    def print_adf_tests_all_assets(self) -> None:
        """
        ADF Stationarity Tests for all assets (prints one table).
        """
        rows = []
        for col in self.returns.columns:
            series = self.returns[col].dropna()
            adf_stat, pval, usedlag, nobs, *_ = adfuller(series)
            rows.append(
                {"asset": col, "ADF_stat": adf_stat, "p_value": pval,
                    "used_lag": usedlag, "n_obs": nobs}
            )

        adf_df = pd.DataFrame(rows).set_index("asset")
        print("\n--- ADF Stationarity Tests (all assets) ---")
        display(adf_df)
        print("------------------------------------------")

    def print_describe_all_assets(self) -> None:
        """
        Descriptive stats for all assets (prints one table).
        """
        desc = self.returns.describe().T
        desc["n_missing"] = self.returns.isna().sum()
        desc["var"] = self.returns.var()

        cols = ["count", "n_missing", "mean", "std",
                "var", "min", "25%", "50%", "75%", "max"]
        desc = desc[cols]

        print("\n--- Descriptive statistics of monthly returns (all assets) ---")
        display(desc)

    # --------------- loop per asset -----------------

    def run_all_per_asset(self, window_vol: int = 24, window_sharpe: int = 24) -> None:
        """
        For each asset, create analysis_plots/<ASSET>/ and save all plots directly there.
        No subfolders.
        """
        for col in self.returns.columns:
            asset_dir = self.base_dir / col
            asset_dir.mkdir(parents=True, exist_ok=True)

            asset_returns = self.returns[[col]]
            dap_asset = DataAnalysisPlots(asset_returns, base_dir=asset_dir)

            dap_asset.plot_boxplots_asset()
            dap_asset.plot_return_distribution_asset()
            dap_asset.plot_rolling_vol_asset(window=window_vol)
            dap_asset.plot_rolling_sharpe_asset(window=window_sharpe)
            dap_asset.plot_drawdowns_asset()
            dap_asset.plot_qq_asset()
