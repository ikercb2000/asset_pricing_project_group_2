# Packages
# --------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from IPython.display import display

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
