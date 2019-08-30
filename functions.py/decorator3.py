#!/usr/bin/env python
# import os
from functools import wraps

sp = '\n\n'

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