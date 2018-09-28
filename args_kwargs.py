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
        total += (i ** a)
    for key, value in kwargs.items():
        print("The item is {} and the quantity is {}".format(key, value))

    for i in kwargs.keys():
        print("The item is {}".format(i))

    print(total)
    return "The total amonunt is ${}".format(total) 


# examples:;
cost = (350, 400, 892, 180)
Inventory = {"Shirts": 350, "Nlow": 400, "knits": 892, "Plood": 180}


addall2(3, 6, 9, 10, shirts=20, Tea=90, belt=10)







