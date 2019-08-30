#!/usr/bin/env python
# import os

def mycounter(func):
    """

    """
    def mywrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        mywrapper.count += 1
        
        return result


    mywrapper.count = 0

    return mywrapper