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

@mytimer
def runtimefunc(n):
    time.sleep(n)
    print(f'sleeping ..... for {n} seconds')


def memoize_arg(func):

    cache = {}

    def wrapper(*args):


        if (args) not in cache:

            time.sleep(3)
            cache[(args)] = func(*args)

        print(cache)
        return cache[(args)]

    return wrapper


def memoize(func2):

    cache = {}

    def wrapper2(*args2, **kwargs):

        value = tuple(val for val in kwargs.values())

        if (args2 + value) not in cache:

            time.sleep(3)
            cache[(args2 + value)] = func2(*args2, **kwargs)

        print(cache)
        return cache[(args2 + value)]

    return wrapper2

@mytimer
@memoize
def longmult(a, b, c, d=8):
    print('sleeping .......')
    return a * b * c * d

longmult(10, 4, 5, d=10)
print('starting second call', end='\n\n')
longmult(10, 10, 5, d=10)
print('starting third call with new args', end='\n\n')
longmult(4, 10, 5, d=10)