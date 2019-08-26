#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, choices
from secrets import choice
import string


def passwordgen(n):
    """Generate password of length n

    Args:
        n (int) : The length of the password

    Returns:
        str
        This is string combination of letters and numbers

    Raises:
        ValueError: if n is not an integer

    """

    # if not isinstance(n, int):
    #     raise ValueError('n must be an integer')
    #     print('n must be an integer')

    # else:
    #     password = ''.join(choice(string.ascii_letters + string.digits) for i in range(n))
    #     return password

    try:
        password = ''.join(choice(string.ascii_letters + string.digits) for i in range(n))
        return password

    except Exception as e: 
        print(e)




passwordgen('ppp')
print(passwordgen(12))