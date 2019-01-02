#!/usr/bin/env python
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler  
from matplotlib import colors
from scipy.ndimage import gaussian_filter

# plt.style.available

# plt.style.use('ggplot')

# # can combine styles in a cascading manner
# plt.style.use(['dark_background', 'classic'])

hh = np.linspace(0,10, 1000)
# # can attach style to particular subplot
# with plt.style.context(('seaborn-deep')):
#     ax1 = plt.subplot2grid((1, 2), (0, 0))
#     ax1.plot(hh, np.sin(hh), 'bo')

# ax2 = plt.subplot2grid((1, 2), (0, 1))
# ax2.hist(np.cos(np.linspace(0,10, 500)))

# plt.pause(2)
# plt.clf()

# # to get the folder for the matplotlib style file
# # make stylelib folder, cod into the foler and
# # make file bigmarker.mplstyle
# mpl.get_configdir()
# plt.style.use('bigmarker')
# plt.hist(np.cos(np.linspace(0,10, 500)))
# plt.pause(2)
# plt.clf()


# plt.imshow(gaussian_filter(np.random.normal(size=(300,300)), sigma=10),
#             cmap='inferno')
# plt.colorbar()

# plt.pause(2)
# plt.clf()


# plt.imshow(gaussian_filter(np.random.normal(size=(300,300)), sigma=10),
#             cmap='inferno')
# plt.colorbar()

# plt.pause(2)
# plt.clf()

# name = 'My_list'
# style_dict = dict(red=[(0, 0, 0.5), (0.5, 1, 0.5), (1, 1, 0.5)],
#                   green=[(0, 0, 1), (1, 0, 1)],
#                   blue=[(0, 0, 1), (1, 0, 1)])

# ccl = colors.ListedColormap(['r', 'b', 'g'])
# ccss = colors.LinearSegmentedColormap(name, style_dict)


# plt.imshow(gaussian_filter(np.random.normal(size=(300,300)), sigma=10),
#             cmap=ccl)
# plt.colorbar()
# plt.pause(2)
# plt.clf()

# plt.imshow(gaussian_filter(np.random.normal(size=(300,300)), sigma=10),
#             cmap=ccss)
# plt.colorbar()
# plt.pause(2)
# plt.clf()



hh = np.arange(0, 10, 0.1)
plt.plot(hh, np.sin(hh), label="1")
plt.plot(hh+1, np.sin(hh), label="2")
plt.plot(hh+2, np.sin(hh), label="3")
plt.plot(hh+3, np.sin(hh), label="4")
plt.plot(hh+4, np.sin(hh), label="5")
plt.plot(hh+5, np.sin(hh), label="6")
plt.legend(loc='best')
plt.pause(4)
plt.clf()


plt.plot(hh, np.sin(hh), label="1")
plt.plot(hh+1, np.sin(hh), label="2")
plt.gca().set_prop_cycle(None)
plt.plot(hh+2, np.sin(hh), label="3")
plt.plot(hh+3, np.sin(hh), label="4")
plt.gca().set_prop_cycle(None)
plt.plot(hh+4, np.sin(hh), label="5")
plt.plot(hh+5, np.sin(hh), label="6")
plt.legend(loc='best')
plt.pause(4)
plt.clf()


cycler1 = (cycler(color=['c', 'm', 'r']))

cylcer2 = (cycler(color=['red', 'blue', 'green'])+
            cycler(linestyle=['-', '-.', '--'])+
            cycler(lw=[1, 2, 3]))
plt.gca().set_prop_cycle(cycler1)

plt.plot(hh, np.sin(hh), label="1")
plt.plot(hh+1, np.sin(hh), label="2")
plt.plot(hh+2, np.sin(hh), label="3")
plt.plot(hh+3, np.sin(hh), label="4")
plt.plot(hh+4, np.sin(hh), label="5")
plt.plot(hh+5, np.sin(hh), label="6")
plt.legend(loc='best')
plt.pause(4)
plt.clf()