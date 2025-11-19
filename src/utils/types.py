# Packages
# --------

from typing import Tuple
from dataclasses import dataclass

# Plotting Portfolios Type
# -------------------------


@dataclass
class plot_ptf:
    mv_pair: Tuple
    color: str = "green"
