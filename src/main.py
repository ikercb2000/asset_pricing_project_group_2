#!/usr/bin/env python
# coding: utf-8

# # Asset Pricing & Risk Group Project: Portfolio Optimization with Dow Jones 30 Stocks
# #### Group 2 : Iker Caballero, Anushka Dhar, Rohan Dinesh, Dylan Ottey
# ---------------------

# ### Package Initialization

# In[10]:


import pandas as pd
import numpy as np
from pathlib import Path


# In[11]:


from utils.visualization import portfolio_sampler, mv_plot
from utils.helpers import ptf_results_print, ann_rets, cov_mat
from utils.optimisers import (
    markowitz_optimiser,
    markowitz_optimiser_oos,
    resampling_optimiser,
    constrained_markowitz_optimiser,
)
from utils.types import plot_ptf
from utils.enums import FreqPrices
from utils.backtesting import run_oos_backtest


# In[12]:


RANDOM_SEED = 123
OPT_PTFS = {}
N_PTFS = 2*10**5
FREQ_DATA = FreqPrices.MONTHLY


# ---

# ### First Step: Compute the Markowitz Efficient Frontier

# The first step is to load the data to the notebook:

# In[13]:


datapath = Path("data","2025-11-17_Dow_Jones_30_Px.csv")
prices = pd.read_csv(datapath, index_col=0)
prices = prices.drop(["DOW","V"],axis=1).replace(0,np.nan).replace(np.inf,np.nan)
monthly_returns = prices.pct_change().dropna()
monthly_returns.head()

# Loading US T-bill monthly data (rf rate)
tbill_path = Path("data", "2025-11-17_GBM_Govt.csv")
tbill = pd.read_csv(tbill_path, parse_dates=["Date"])

# Setting index and align to stock returns period
tbill = tbill.set_index("Date").sort_index()
tbill = tbill.loc[monthly_returns.index.min(): monthly_returns.index.max()]

# Ensuring PX_LAST is numeric and Converting from percent to decimal
tbill["PX_LAST"] = pd.to_numeric(tbill["PX_LAST"], errors = "coerce")
tbill["rf_ann"] = tbill["PX_LAST"]/100.0

# Dropping any rows where rf_ann could not be computed
tbill = tbill.dropna(subset = ["rf_ann"])

# Computing average annual T-bill rate for the entire OOS period
rf_ann = tbill["rf_ann"].mean()

print(f"Using average annual T-bill rf = {rf_ann:.4f}")


# From the data, compute the mean vector and the covariance matrix of the data, annualized:

# In[14]:


mu_returns: pd.Series = ann_rets(monthly_returns,FREQ_DATA)
cov_returns: pd.DataFrame = cov_mat(monthly_returns,FREQ_DATA)
n_obs: int = monthly_returns.shape[0]


# The Markowitz Optimal Portfolio given the data is the following:

# In[15]:


weights, opt_ret, opt_vol = markowitz_optimiser(mu_ret=mu_returns, cov_ret=cov_returns)
orig_markowitz_opt_ptf = (opt_ret,opt_vol**2)


# In[16]:


OPT_PTFS["orig_markowitz"] = plot_ptf(mv_pair=orig_markowitz_opt_ptf,color = "green")
ptf_results_print(opt_ret, opt_vol, weights)


# Now we can get an overview of the mean-variance frontier and visualize where the optimal Markowitz portfolio is:

# In[17]:


mv_pairs, _, _ = portfolio_sampler(mu_rets=mu_returns, cov_rets=cov_returns, n_portfolios=N_PTFS, min_assets=1)
mv_plot(mv_pairs, "plots/efficient_frontier.png", highlight_ptf=OPT_PTFS)


# ---

# ### Second Step: Compute Optimal Portfolio by Resampling

# Blah blah blah...

# In[23]:


weights, opt_ret, opt_vol = resampling_optimiser(mu_ret=mu_returns, cov_ret=cov_returns, frequency=FREQ_DATA, n_obs= n_obs, random_state=RANDOM_SEED, n_bootstrap=1000)
resampling_opt_ptf = (opt_ret,opt_vol**2)


# In[25]:


OPT_PTFS["resampling_markowitz"] = plot_ptf(mv_pair=resampling_opt_ptf,color="orange")
ptf_results_print(opt_ret, opt_vol, weights)


# In[20]:


mv_pairs, _, _ = portfolio_sampler(mu_rets = mu_returns,cov_rets = cov_returns, n_portfolios=N_PTFS, min_assets=1)
mv_plot(mv_pairs, "plots/resampling_efficient_frontier.png", highlight_ptf=OPT_PTFS)


# ---
# ### Third Step: Out-of-Sample Rolling_Window Backtest
# ---

# Rolling window length (in months)
WINDOW = 60

MV_PLOT_DIR = Path("plots", "mv_oos")
MV_PLOT_DIR.mkdir(parents = True, exist_ok = True)

methods = {
    "Markowitz": markowitz_optimiser_oos,
    "Constrained (10%)": constrained_markowitz_optimiser,
}

oos_df, stats_df = run_oos_backtest(
    returns=monthly_returns,
    frequency=FREQ_DATA,
    window=WINDOW,
    methods=methods,
    mv_plot_dir=MV_PLOT_DIR,
    show_plots=False,
    do_plots=False,
    print_res=False,
    n_ptfs=1000,
    min_assets=1,
    max_assets=None,
    random_state=RANDOM_SEED,
    rf=rf_ann,
)

print("\n=== OOS Performance Statistics ===")
print(stats_df)

# ---

# **Notes for Iker (but use them if you want to motivate yourselves on how to improve your sections):**
# 
# - Add graphs as in the papers, that would create a good analysis of the methodologies and results
# - Add bar charts for portfolio weights (like in paper)
# - Add a graph with optimal resampling B, using MSE (variance/return/sharpe ratio) criteria
# - At the end of the implementations, produce a comparitive table between each other in Pandas
# - Add graphs and tables for comparing with benchmarks
# - Modify/finish Readme.md
# - Graph of realized Information Ratios (boxes) of Ledoit(2003a)
