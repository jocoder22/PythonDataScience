#  *args for passing non-keyword list of argument to a function;
# this makes the *args iterable and optional arguments
# *args can accept list as argument


def addall(a, *args):
    total = 0
    for i in args:
        total += i
    return total

    
# **kwargs for passing keyword list ( or dictionary) to a function
# This makes **kwargs iterable, optional and can accept dictionary as argument;
