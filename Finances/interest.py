#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def interest(amt, rate, timet):
    int = (1 + (rate/12)) ** timet
    total = amt * int 
    return total-amt, total 

a, b = interest(46000, 0.03, 48)
print(f'Interest paid is ${a:.2f}, and total amount is ${b:.2f}')


a, b = interest(320000, 0.04, 360)
print(f'Interest paid is ${a:.2f}, and total amount is ${b:.2f}')
print(f'Monthly payment is {b/360:.2f}')


a, b = interest(320000, 0.03, 360)
print(f'Interest paid is ${a:.2f}, and total amount is ${b:.2f}')
print(f'Monthly payment is {b/360:.2f}')
