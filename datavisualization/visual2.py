#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt

# plt.ion()

x1 = np.arange(0, 4, 0.1)
plt.plot(x1, x1, label='Linear')
plt.plot(x1, np.sin(x1), label='sin(x)')
plt.plot(x1, x1**2, label='Quadratic')
plt.plot(x1, np.cos(x1), label='cos(x)')


plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('My graphic display I')
plt.legend()

plt.show()
plt.close()

# Ploting multiple plots: using subplot
groups = ['group1', 'group2', 'group3']
sales = [590, 800, 5600]
members = [13, 25, 100]
totalHr = [214, 689, 3280]

plt.figure(1, figsize=(10.5, 4), facecolor='green',
           edgecolor='r')
plt.subplot(231)
plt.bar(groups, sales)
plt.subplot(232)
plt.scatter(groups, members)
plt.subplot(233)
plt.plot(groups, totalHr, 'r^')
plt.subplot(234)
plt.plot(x1, x1**3, color='orange', linestyle='-.')
plt.subplot(235)
plt.plot(x1, x1**4, color='cyan', linestyle=':', marker='.')
plt.subplot(236)
plt.plot(x1, x1**5)

plt.show()