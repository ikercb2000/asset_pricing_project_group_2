#!/usr/bin/env python
# coding: utf-8

# # Asset Pricing & Risk Group Project: Portfolio Optimization with Dow Jones 30 Stocks
# #### Group 2 : Iker Caballero, Anushka Dhar, Rohan Dinesh, Dylan Ottey
# ---------------------

# ### Package Initialization

# In[13]:


import pandas as pd
import numpy as np
import glob

from pathlib import Path
from PIL import Image
from IPython.display import Image as IPImage, display


# In[14]:


from utils.data_analysis import DataAnalysisPlots
from utils.optimisers import markowitz_optrisky_ptf, equally_weighted_ptf, resampling_optimiser, constrained_markowitz_optimiser, shrinkage_markowitz_optimiser
from utils.enums import FreqPrices
from utils.backtesting import run_oos_backtest


# ### Specify Parameters

# In[15]:


RANDOM_SEED = 123
OPT_PTFS = {}
N_PTFS = 10**4
N_BOOTSTRAP = 100
FREQ_DATA = FreqPrices.MONTHLY


# ### Specify Output Directories

# In[16]:


Path("..","plots").mkdir(parents=True, exist_ok=True)
analysis_dir = Path("..","plots","data_analysis")
analysis_dir.mkdir(parents=True, exist_ok=True)
mv_plots_path = Path("..","plots", "optimal_ptfs")
mv_plots_path.mkdir(parents=True, exist_ok=True)
alloc_plots_path = Path("..","plots","allocation_plots")
alloc_plots_path.mkdir(parents=True, exist_ok=True)


# ---

# ### First Step: Data Analysis

# The first step is to load the data to the notebook. We obtained the data from the Bloomberg terminal in .csv format, and hence we load the adjusted closed prices:

# In[25]:


datapath = Path("..", "data", "2025-11-17_Dow_Jones_30_Px.csv")
prices = pd.read_csv(datapath,parse_dates=["Date"],index_col="Date")
monthly_returns = prices.pct_change()
monthly_returns = monthly_returns.replace(np.inf, np.nan).iloc[3:, :]
monthly_returns.head()


# In[ ]:


rf_path = Path("..","data","2025-11-17_GBM_Govt_to_send.csv")
tbill = pd.read_csv(rf_path, index_col=0)
tbill.index = pd.to_datetime(tbill.index, unit="D", origin="1899-12-30")
tbill = tbill.sort_index()
y_ann = tbill["PX_LAST"] / 100.0
rf_monthly = (1 + y_ann) ** (1/12) - 1
rf_monthly = rf_monthly.reindex(monthly_returns.index)
rf_monthly = rf_monthly.fillna(method="ffill").astype(float)
print(rf_monthly)


# From the price data, we can obtain the arithmetic returns for each month. We need to erase the first row (no returns for the initial price date), but we let **NaN** values in order to provide more analysis. The resulting dataset is the following:

# In[ ]:


dap = DataAnalysisPlots(monthly_returns, base_dir=analysis_dir)


# Now we provide an statistical analysis of the data to have a better understanding of the returns.

# In[20]:


dap = DataAnalysisPlots(monthly_returns, base_dir=analysis_dir)

dap.plot_corr_heatmap()
dap.plot_boxplots_all_assets()
dap.plot_return_distributions_all_assets(max_assets=6)
dap.plot_rolling_corr_all_assets(window=36)
dap.plot_pca_variance()
dap.print_adf_tests_all_assets()
dap.print_describe_all_assets()
dap.run_all_per_asset(window_vol=12, window_sharpe=24)


# From the rolling volatility plot, we can see that there are periods with high market risk (such as the 2008 crisis or the Covid crisis), so that we can expect that asset risk is highly time-varying, which would affect the optimal portfolios depending on the batch.However, the correlation heatmap reveals medium and heterogeneous correlations across the stoocks, suggesting that diversification is achievable and expected when using tangency portfolios. Moreover, the boxplots show heavy tails, outliers, and non-normal return distributions, implying unstable estimates of means and covariances when applying the different portfolio optimisation techniques.
# 
# Therefore, we can claim that the data exhibits typical stock return characteristics such as volatility clustering, imperfect correlations, and non-Gaussian behavior, which highlights the challenges of classical Markowitz estimation and motivates the use of robust techniques such as the ones we will try.

# ---

# ### Second Step: Backtest Optimal Portfolios Using Different Methods

# Now that we have seen how our data is and its characteristics, we can now apply some portfolio optimization techniques in order to test the out-of-sample performance when choosing an optimal risky portfolio. In this case, we have chosen to use the Resampling Portfolio Optimiser appearing in Michaud (????) and Schner (????) and the Constrained Portfolio Optimiser from AUTHOR (????). We will use the original Markowitz Risky Portfolio Optimiser and the 1/N-Portfolio as a benchmarks.

# In[21]:


methods = {"Markowitz": lambda mu, cov, n_obs: markowitz_optrisky_ptf(mu_ret=mu, cov_ret=cov),
           "1/N-Ptf": lambda mu, cov, n_obs: equally_weighted_ptf(mu_ret=mu, cov_ret=cov),
           "Resampling": lambda mu, cov, n_obs: resampling_optimiser(mu_ret=mu,cov_ret=cov,n_obs=n_obs,
                                                              random_state=RANDOM_SEED,n_bootstrap=N_BOOTSTRAP),
           "Constrained": lambda mu, cov, n_obs: constrained_markowitz_optimiser(mu_ret=mu, cov_ret=cov, max_weight= 0.1),
           "Shrinkage_0.66":   lambda mu, cov, n_obs: shrinkage_markowitz_optimiser(mu_ret=mu,cov_ret=cov,n_obs=n_obs,shrinkage=0.66),
           "Shrinkage_0.33":   lambda mu, cov, n_obs: shrinkage_markowitz_optimiser(mu_ret=mu,cov_ret=cov,n_obs=n_obs,shrinkage=0.33)}


# Once we have defined the optimisers or functions that will return the weights of the portfolio for each asset, we run a backtest computing a moving window with length of 60 monthly returns in order to test the (historical) out-of-sample performance of the different portfolios produced with these methods.

# In[22]:


oos_df, stats_df = run_oos_backtest(returns=monthly_returns, frequency=FreqPrices.MONTHLY, window=60,
    methods=methods, mv_plot_dir=mv_plots_path, show_plots=False, do_plots=True, print_res=False, n_ptfs=N_PTFS, rf_series=rf_monthly, alloc_plot_dir=alloc_plots_path)


# Once the Backtest is finalised, we can now show the results and comment them.

# In[ ]:


print("\n----------- Final Results -----------\n\n",stats_df)

gif_path = Path(mv_plots_path,"optimal_ptfs_evolution.gif")

frames = [Image.open(img) for img in sorted(glob.glob(f"{mv_plots_path}/*.png"))]
frames[0].save(gif_path,save_all=True,append_images=frames[1:],duration=300,loop=0)

gif_alloc_path = Path(alloc_plots_path, "allocations_evolution.gif")

frames = [Image.open(img) for img in sorted(glob.glob(f"{alloc_plots_path}/*.png"))]
frames[0].save(
    gif_alloc_path,
    save_all=True,
    append_images=frames[1:],
    duration=300,
    loop=0,
)

print("\nThis is a GIF of the evolution of portfolio weights:\n")
display(IPImage(filename=str(gif_alloc_path)))

print("\n","This is a GIF animation of the portfolios evolution compared to the efficient frontier:","\n")
display(IPImage(filename=str(gif_path)))


# ---

# **Notes for Iker (but use them if you want to motivate yourselves on how to improve your sections):**
# - Add a graph with optimal resampling B, using MSE (variance/return/sharpe ratio) criteria
# - Add graphs and tables for comparing with benchmarks
# - Modify/finish Readme.md
# - Graph of realized Information Ratios (boxes) of Ledoit(2003a)
