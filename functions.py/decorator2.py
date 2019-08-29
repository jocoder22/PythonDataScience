#!/usr/bin/env python
# import os

import time


import time
def mytimer(func):
    """Calculate time in seconds to run a program

    Args:
        func: Function
        The function to estimate the run time

    Retuns:
        Function

    """
    def wrapper(*args, **kwargs):

        # take the start time
        start_time = time.time()

        # run the decorated function
        result = func(*args, **kwargs)

        # take the finish time
        finish_time = time.time()

        # find the elapse time
        duration = finish_time - start_time

        print(f'{func.__name__} took {duration} seconds')

        return result

    return wrapper