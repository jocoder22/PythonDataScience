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

        print(f'{func.__name__} took {duration} seconds', end='\n\n')

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

        value = tuple(f'{val}{kwargs[val]}' for val in sorted(kwargs))
        value2 = tuple(sorted(kwargs.items(), key = lambda kv_pair:(kv_pair[0], kv_pair[1])))

        if (args2 + value) not in cache:

            time.sleep(3)
            cache[(args2 + value)] = func2(*args2, **kwargs)

        print(cache)

        return cache[(args2 + value)]

    return wrapper2

@mytimer
@memoize_arg
def longmult(a, b, c):
    print('sleeping .......')
    return a * b * c 

longmult(10, 4, 5)
print('starting second call for decorator without kwargs')
longmult(10, 4, 5)
print('starting third call for decorator without kwargs with new args')
longmult(4, 10, 5)



@mytimer
@memoize
def longmult2(a, b, c, **kwargs):
    print('sleeping .......')
    result = a * b * c 
    for k in kwargs.values():
        result *= k
    return result


longmult2(10, 4, 5)
print('starting second call')
longmult2(10, 4, 5, d=10)
print('starting third call with new args')
longmult2(4, 10, 5, d=10)
print('starting fourth call with 2 kwargs')
longmult2(10, 4, 5, d=10, e=10)
print('starting fifth call with 2 kwargs interchanged')
longmult2(10, 4, 5, e=10, d=10)
print('starting sixth call with  args and 2 kwargs interchanged')
longmult2(4, 10, 5, e=10, d=10)