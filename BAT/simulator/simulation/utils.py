import numpy as np


def price2bin(price: float) -> int:
    """
    Conversion from BID value to BIN value
    price = 1.2 ** bin
    """
    if price <= 0:
        return 0
    return np.round(np.log(price) / np.log(1.2))


def bin2price(bin_: int) -> float:
    """
    Reverse conversion from BIN value to BID value
    price = 1.2 ** bin
    """
    return np.power(1.2, bin_)
