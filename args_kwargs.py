#  *args for passing non-keyword list of argument to a function;
# this makes the *args iterable;
def addall(a, *args):
    total = 0
    for i in args:
        total += i
    return total

    
