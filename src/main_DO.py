import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, plotting


FREQ_MONTH = 12

issue_dates = {
    'DOW': '2019-03-29',
    'V': '2008-03-31'
}

# TO-DO:

# Convert pd based on index of first column.... AFFECTING EVERYTHING HOW WILL THINGS NOW CORRECTION SLICE?


# Created separate file to for on Markowitz Efficient Frontier, Markowitz mean-variance portfolio
# based on sample estimates and Naive equally-weighted portfolio
# Can combine everything after our meeting


# MARKOWITZ EFFICIENT FRONTIER

# Import Price and Risk-Free Data

Dow_30 = pd.read_csv("data/2025-11-17_Dow_Jones_30_Px.csv")
GBM_Govt = pd.read_csv("data/2025-11-17_GBM_Govt.csv")


# Covert dataframes so that we are working with the same time period

Dow_30 = Dow_30[Dow_30["Date"].isin(GBM_Govt["Date"])]
Dow_30 = Dow_30.reset_index(drop=True)

# Set_Index for "Date" column
Dow_30 = Dow_30.set_index("Date")
GBM_Govt = GBM_Govt.set_index("Date")


# Covert Dow_30_Px to Dow_30_Return

def per_period_returns(price_data):
    '''Convert a dataframe of prices into return per period'''
    prices = price_data.copy()
    returns = prices.pct_change()
    returns = returns[1:].reset_index().set_index("Date")
    returns = returns.replace(
        [np.nan, np.inf, -np.inf], 0)
    return returns


Dow_30_returns = per_period_returns(Dow_30)
print(Dow_30_returns)

# Convert GBM_Govt prices into percentages

GBM_Govt = GBM_Govt/100

# CHECK: Need to convert T-bill rates to monthly to match terms as BBG annualises


def periodic_rates(risk_free_data, periods):
    rates = risk_free_data.copy()
    rates = (1 + rates) ** (1/periods) - 1
    return rates


# Set GBM_Govt_monthly to same index as Dow_30_returns
GBM_Govt_monthly = periodic_rates(GBM_Govt, FREQ_MONTH)
GBM_Govt_monthly = GBM_Govt_monthly[1:].reset_index().set_index("Date")
print(GBM_Govt_monthly)


# Calculate the monthly excess returns

def excess_returns(return_data, risk_free_data):
    ''' Takes price_data and rates data and return the excess returns over
    the whole period '''
    returns = return_data.copy()
    rates = risk_free_data.copy()
    num_risky_assets = returns.shape[1]

    print(returns.shape)
    print(rates.shape)

    for column in range(num_risky_assets):

        if returns.columns[column] in issue_dates:
            start_date = issue_dates.get(returns.columns[column])
            start_date_index = returns.index.get_loc(start_date)
            returns.iloc[start_date_index:, column] = returns.iloc[start_date_index:,
                                                                   column] - rates.iloc[start_date_index:, 0]

        else:
            returns.iloc[:, column] = returns.iloc[:,
                                                   column] - rates.iloc[:, 0]

    return returns


Dow_30_excess_returns = excess_returns(Dow_30_returns, GBM_Govt_monthly)
# print(Dow_30_excess_returns)


# Test CSV to check the underlying data passed a sanity check

# Dow_30_returns.to_csv("data/DO_test_returns.csv", index=False)
# Dow_30_excess_returns.to_csv("data/DO_test_excess.csv", index=False)


# Expected Excess Returns on an annualised basis (whole period)

def expected_return(return_data):
    returns = return_data.copy()
    return returns.mean()


Dow_30_expected_exc_returns = expected_return(Dow_30_excess_returns)
print(Dow_30_expected_exc_returns)

# Annualise our expected excess returns


def annualise_return(return_data, periods):
    returns = return_data.copy()
    returns = (1 + returns) ** periods - 1
    return returns


Dow_30_exp_exc_returns_ann = annualise_return(
    Dow_30_expected_exc_returns, FREQ_MONTH)

# print(Dow_30_exp_exc_returns_ann)


# Variance-Covariance Matrix

def cov_matrix(return_data):
    returns = return_data.copy()
    monthly_cov_matrix = returns.cov()
    annual_cov_matrix = FREQ_MONTH * monthly_cov_matrix
    return annual_cov_matrix


Dow_30_cov_matrix = cov_matrix(Dow_30_returns)
# print(Dow_30_cov_matrix)


# Constructing the Markovwitz Efficiency Frontier

# marko_eff_frontier = EfficientFrontier(
#     Dow_30_exp_exc_returns_ann, Dow_30_cov_matrix)


# fig, ax = plt.subplots()
# plotting.plot_efficient_frontier(marko_eff_frontier, ax=ax)
# plt.show()

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # IN-SAMPLE MEAN-VARIANCE BENCHMARK TANGENCY PORTFOLIO

def inverse_matrix(matrix):
    regularisation = 10**-6
    matrix_np = matrix.to_numpy()
    matrix_np_var = np.diag(matrix_np)
    zero_var_check = np.any(matrix_np_var < 10**-10)
    if zero_var_check:
        reg_matrix_np = matrix_np + regularisation * np.eye(len(matrix_np))
        return np.linalg.inv(reg_matrix_np)
    else:
        return np.linalg.inv(matrix_np)


def mean_var_sample_weight(expected_ann_return_data, cov_matrix):
    ones_n_risky = np.ones(len(expected_ann_return_data))
    expected_return_data_np = expected_ann_return_data.to_numpy()
    inv_cov_matrix = inverse_matrix(cov_matrix)
    weights = (inv_cov_matrix @ expected_return_data_np) / \
        (ones_n_risky @ inv_cov_matrix @ expected_return_data_np)
    return weights


W_IS = mean_var_sample_weight(Dow_30_exp_exc_returns_ann, Dow_30_cov_matrix)

# print(W_IS)


# Calculate benchmark returns

benchmark_returns = Dow_30_excess_returns @ W_IS
print(benchmark_returns)


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # MARKOWITZ MEAN-VARIANCE PORTFOLIO BASED ON SAMPLE ESTIMATES

# # Rolling window

M = 60


def rolling_window(return_data, window):
    return_data_all = return_data.copy()
    return_data_rolling = [return_data_all.iloc[i-window: i]
                           for i in range(window, len(return_data_all))]
    return return_data_rolling


Dow_30_excess_returns_rolling = rolling_window(Dow_30_excess_returns, M)

# print(Dow_30_excess_returns_rolling[0])


def portfolio_mean_variance_sample(return_data_rolling):
    return_data_rolling = return_data_rolling.copy()
    weights = []
    for i, return_data_window in enumerate(return_data_rolling):
        sample_cov_matrix = cov_matrix(return_data_window)
        sample_expected_return = expected_return(return_data_window)
        sample_annualise_return = annualise_return(
            sample_expected_return, FREQ_MONTH)
        sample_weights = mean_var_sample_weight(
            sample_annualise_return, sample_cov_matrix)
        weights.append(sample_weights)
    return weights


mean_variance_sample_weights = portfolio_mean_variance_sample(
    Dow_30_excess_returns_rolling)

# print(len(testing))

# for i in testing:
#     sum = np.sum(i)
#     if abs(sum - 1) > 1e-6:
#         print("Does not sum to 1")
#     else:
#         print("Fine")


benchmark_returns_sample = []
for i in range(M, len(Dow_30_excess_returns)):
    benchmark_returns_sample.append(
        Dow_30_excess_returns.iloc[i] @ mean_variance_sample_weights[i-M])

benchmark_returns_sample = pd.Series(
    benchmark_returns_sample, index=Dow_30_excess_returns.index[M:])

print(benchmark_returns_sample.tail())


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # NAIVE EQUALLY-WEIGHTED PORTFOLIO

# # 1/N weighting across all assets (when non-zero)

N_risky_assets = (Dow_30_excess_returns[M:] != 0).sum(axis=1)
# print(N_risky_assets)

Dow_30_excess_returns_naive = Dow_30_excess_returns.iloc[M:].div(
    N_risky_assets, axis=0)
# print(Dow_30_excess_returns_naive)

benchmark_returns_naive = Dow_30_excess_returns_naive.sum(axis=1)
# print(benchmark_returns_naive)


# Returns Sanity Checks - THINK THERE COULD BE ERRORS

print(benchmark_returns.mean(), benchmark_returns_naive.mean(),
      benchmark_returns_sample.mean())

compound_benchmark_returns = (1 + benchmark_returns).prod()-1
compound_benchmark_returns_naive = (1 + benchmark_returns_naive).prod()-1
compound_benchmark_returns_sample = (1 + benchmark_returns_sample).prod()-1

print(compound_benchmark_returns, compound_benchmark_returns_naive,
      compound_benchmark_returns_sample)
