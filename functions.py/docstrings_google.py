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

    if not isinstance(n, int):
        print('n must be an integer')

    print('password')



passwordgen('ppp')