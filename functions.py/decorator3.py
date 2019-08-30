#!/usr/bin/env python
# import os

def mycounter(func):
    """

    """
    def mywrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        mywrapper.count += 1

        print(f'The square of {(args, kwargs)} is {result}')


    mywrapper.count = 0

    return mywrapper

@mycounter
def squarefunc(n):
    
    print(f'Called {squarefunc.__name__} with {n} argument!')

    return n ** 2


squarefunc(5)
squarefunc(4)

print(f'Called {squarefunc.__name__} {squarefunc.count} times')