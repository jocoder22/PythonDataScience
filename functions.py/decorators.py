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


