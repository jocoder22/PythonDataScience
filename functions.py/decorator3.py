#!/usr/bin/env python
# import os
from functools import wraps

def print2(*args):
    for arg in args:
        print(arg, end='\n\n') 

sp = {"sep":"\n\n", "end":"\n\n"}  

def mycounter(func):
    """

    """
    @wraps(func)
    def mywrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        mywrapper.count += 1

        print(f'The square of {(args, kwargs)} is {result}')

        return result


    mywrapper.count = 0

    return mywrapper

@mycounter
def square(n=1):
    """This return the square of a number

    Args: int
        The number to find the square

    Returns: int
    """
    print(f'Called {square.__name__} function with {n} argument!')

    return n ** 2


# square(5)
# square(4)

print(f'Called {square.__name__} function {square.count} times')

print(square.__doc__, square.__defaults__, square.__wrapped__, sep=sp, end=sp)

# Accessing the original undecorated function using originalfunction.__wrapped__
print(square.__wrapped__(6), end=sp)
print(square.__wrapped__.__doc__, end=sp)