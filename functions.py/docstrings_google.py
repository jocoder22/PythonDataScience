#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, choices
from secrets import choice
import string
import inspect



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

    try:
        password = ''.join(choice(string.ascii_letters + string.digits) for i in range(n))
        return password

    except Exception as e: 
        print(f'{e}, argument must be an integer.')


passwordgen(12.0)
print(passwordgen(12))

# print the docstring
print(passwordgen.__doc__)
print(inspect.getdoc(passwordgen))