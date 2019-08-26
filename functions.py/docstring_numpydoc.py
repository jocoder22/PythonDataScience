#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, choices
from secrets import choice
import string
import inspect




def printdocstring(func):
    """Prints out the docstring of functions

    Parameters
    ----------
    func: object
        The function that need to print the docstring

    Returns
    --------
        str:
        The function docstring

    """

    print(inspect.getdoc(func))

printdocstring(np.amax)