#!/usr/bin/env python
# import os

def parentFunct(s):
    """

    """
    # Nonlocal variables
    bet = 90
    bet_draw = bet * 1000 / bet ** 2
    mydict = dict({'first': 5869, 'second':346})
    mylist = list([4905, 9403, 560])


    def childFunct():
        print(bet)
        print(mylist)
        print(mydict)
        return bet_draw
    
    return childFunct



myfunc = parentFunct(34)

print(myfunc())

# closures
print(type(myfunc.__closure__))
print(len(myfunc.__closure__))
closure_values = [myfunc.__closure__[i].cell_contents for i in range(len(myfunc.__closure__))]
print(closure_values)

print([cell.cell_contents for cell in myfunc.__closure__])
print(myfunc.__closure__)
print(myfunc.__closure__[0])
print(myfunc.__closure__[2].cell_contents)
