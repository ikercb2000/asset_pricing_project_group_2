# Packages
# --------

import pandas as pd

# Annualized Means
# ----------------


def ann_rets(returns: pd.DataFrame) -> pd.DataFrame:
    """Obtains mean vector of annualized returns"""
    return (1+returns.mean())**252-1

# Covariance Matrix
# -----------------


def cov_mat(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Obtains covariance matrix of annualized returns
    """

    return returns.cov()*252
