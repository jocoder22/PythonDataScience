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

def addall2(a, *agrs, **kwargs):
    total = 0
    for i in agrs:
        total += (i ** 3)
    for key, value in kwargs.items():
        print("The key is {} and the value is {}".format(key, value))
    
    for i in kwargs.keys():
        print("The key is {}".format(i))
    
    print(total)
    return "The total amonunt is ${}".format(total) 


