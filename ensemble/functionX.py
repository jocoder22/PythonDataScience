
import os
from contextlib import contextmanager

@contextmanager
def changepath(path):
    currentpath = os.getcwd()

    os.chdir(path)

    try:
        yield 

    finally:
        os.chdir(currentpath)
