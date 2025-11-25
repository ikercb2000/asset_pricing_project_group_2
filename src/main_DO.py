import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, plotting

# TO-DO:

# Convert pd based on index of first column....


# Created separate file to for on Markowitz Efficient Frontier, Markowitz mean-variance portfolio
# based on sample estimates and Naive equally-weighted portfolio
# Can combine everything after our meeting


# MARKOWITZ EFFICIENT FRONTIER

# Import Price and Risk-Free Data

Dow_30 = pd.read_csv("data/2025-11-17_Dow_Jones_30_Px.csv")
GBM_Govt = pd.read_csv("data/2025-11-17_GBM_Govt.csv")


issue_dates = {
    'DOW': '2019-03-29',
    'V': '2008-03-31'
}

# Covert dataframes so that we are working with the same time period

Dow_30 = Dow_30[Dow_30["Date"].isin(GBM_Govt["Date"])]
Dow_30 = Dow_30.reset_index(drop=True)


# Covert Dow_30_Px to Dow_30_Return_

def per_period_returns(price_data):
    '''Convert a dataframe of prices into return per period'''
    prices = price_data.copy()
    prices.iloc[:, 1:] = prices.iloc[:, 1:].pct_change()
    prices = prices[1:]
    prices.iloc[:, 1:] = prices.iloc[:, 1:].replace(
        [np.nan, np.inf, -np.inf], 0)
    return prices


Dow_30_returns = per_period_returns(Dow_30)

# Convert GBM_Govt prices into percentages

GBM_Govt.iloc[:, 1:] = GBM_Govt.iloc[:, 1:]/100


# CHECK: Need to convert T-bill rates to monthly to match terms as BBG annualises

def periodic_rates(risk_free_data, periods):
    rates = risk_free_data.copy()
    rates.iloc[:, 1:] = (1 + rates.iloc[:, 1:]) ** (1/periods) - 1
    return rates


GBM_Govt_monthly = periodic_rates(GBM_Govt, 12)


# Calculate the monthly excess returns

def excess_returns(return_data, risk_free_data):
    ''' Takes price_data and rates data and return the excess returns over 
    the whole period '''
    returns = return_data.copy()
    rates = risk_free_data.copy()
    rates = rates.iloc[1:,]

    for column in range(1, returns.shape[1]):

        if returns.columns[column] in issue_dates:
            start_date = issue_dates.get(returns.columns[column])
            start_date_index = returns.index[
                returns["Date"] == start_date][0]
            returns.iloc[start_date_index:, column] = returns.iloc[start_date_index:,
                                                                   column] - rates.iloc[start_date_index:, 1]

        else:
            returns.iloc[:, column] = returns.iloc[:,
                                                   column] - rates.iloc[:, 1]

    return returns


Dow_30_excess_returns = excess_returns(Dow_30_returns, GBM_Govt_monthly)


# Test CSV to check the underlying data passed a sanity check

# Dow_30_returns.to_csv("data/DO_test_returns.csv", index=False)
# Dow_30_excess_returns.to_csv("data/DO_test_excess.csv", index=False)


# Expected Excess Returns on an annualised basis (whole period)

def expected_return(return_data):
    returns = return_data.copy()
    return returns.iloc[:, 1:].mean()


Dow_30_expected_exc_returns = expected_return(Dow_30_excess_returns)


# Annualise our expected excess returns
def annualise_return(return_data, periods):
    returns = return_data.copy()
    returns = (1 + returns) ** periods - 1
    return returns


Dow_30_exp_exc_returns_ann = annualise_return(Dow_30_expected_exc_returns, 12)

print(Dow_30_exp_exc_returns_ann)


# Variance-Covariance Matrix


def cov_matrix(return_data):
    returns = return_data.copy()
    monthly_cov_matrix = returns.iloc[:, 1:].cov()
    annual_cov_matrix = 12 * monthly_cov_matrix
    return annual_cov_matrix


Dow_30_cov_matrix = cov_matrix(Dow_30_returns)


# Constructing the Markovwitz Efficiency Frontier

# marko_eff_frontier = EfficientFrontier(
#     Dow_30_exp_exc_returns_ann, Dow_30_cov_matrix)


# fig, ax = plt.subplots()
# plotting.plot_efficient_frontier(marko_eff_frontier, ax=ax)
# plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# IN-SAMPLE MEAN-VARIANCE BENCHMARK TANGENCY PORTFOLIO

no_risky_assets = 30

ones_n_risky = np.ones(no_risky_assets)

Dow_30_exp_exc_return_ann_np = Dow_30_exp_exc_returns_ann.to_numpy()


def inverse_matrix(matrix):
    matrix_np = matrix.to_numpy()
    return np.linalg.inv(matrix_np)


inv_cov_matrix = inverse_matrix(Dow_30_cov_matrix)


W_IS = (inv_cov_matrix @ Dow_30_exp_exc_returns_ann) / \
    (ones_n_risky @ inv_cov_matrix @ Dow_30_exp_exc_returns_ann)


# Calculate benchmark returns

# CHANGE THIS WHEN YOU FIX THE REST OF THE FILE
Dow_30_excess_returns = Dow_30_excess_returns.set_index("Date")

benchmark_returns = Dow_30_excess_returns @ W_IS


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# MARKOWITZ MEAN-VARIANCE PORTFOLIO BASED ON SAMPLE ESTIMATES


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# NAIVE EQUALLY-WEIGHTED PORTFOLIO
