# Packages
# --------

from enum import Enum

# Frequency Enum
# --------------


class FreqPrices(Enum):
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    YEARLY = 1
