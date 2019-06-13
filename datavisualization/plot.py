#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# plt.ion()

xplot = np.linspace(0, 10, 1000)
yplot = np.sin(xplot)

plt.plot(xplot, yplot)
plt.show()

'''
Colors:

'b'	blue
'g'	green
'r'	red
'c'	cyan
'm'	magenta
'y'	yellow
'k'	black
'w'	white
'''

plt.plot(xplot, yplot, color='r')


"""
Linestyle:

'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
"""

plt.plot(xplot+2, yplot, color='b', linestyle=':')
plt.plot(xplot+3, yplot, color='m', linestyle='-.')
plt.show()

'''
Markers:

'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
'1'	tri_down marker
'2'	tri_up marker
'3'	tri_left marker
'4'	tri_right marker
's'	square marker
'p'	pentagon marker
'*'	star marker
'h'	hexagon1 marker
'H'	hexagon2 marker
'+'	plus marker
'x'	x marker
'D'	diamond marker
'd'	thin_diamond marker
'|'	vline marker
'_'	hline marker

'''

plt.plot(xplot, yplot, color='g', marker='>', markevery=10)

plt.plot(xplot+2, yplot, color='r', marker='^', markevery=10)

plt.plot(xplot+2.5, yplot, color='b', linestyle=':', marker='^', markevery=10)
plt.show()

# markersize affects the markers, while linewith affects the linestyle
plt.plot(xplot+5, yplot, color='g', marker='p', markersize=0.5,
         linewidth=3, markevery=10)
plt.axis('off')
plt.show()
# zorder is the ordering of overlayed plot, higher number are in front
# alpha affects transparency of the plot
plt.plot(xplot+3, yplot, color='b', linewidth=10, zorder=8, alpha=0.8)
plt.plot(xplot, yplot, color='g', linewidth=10, zorder=15, alpha=0.3)
plt.plot(xplot+7, yplot, color='g', linewidth=10, zorder=15, alpha=0.3)

# Remember to close the pyplot
plt.close()
