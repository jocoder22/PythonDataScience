# !/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


xplot = np.linspace(0, 10, 1000)
yplot = np.sin(xplot)


mylist = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn']
'''
          'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
          'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
          'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired']
         
          'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
          'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
          'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
          'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
          'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
          'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
          'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',
          'YlOrRd', 'YlOrRd_r']
'''

for i in mylist:
    plt.title('This is for cmap: ' + i)
    plt.scatter(xplot, yplot, c=np.cos(xplot), cmap=i,
                edgecolors='none',
                s=np.power(xplot, 4))
    plt.pause(1)
    plt.cla()

plt.close()

# # t = 0
# # while t < len(mylist):
# #     plt.title('This is for cmap: ' + mylist[t])
# #     plt.scatter(xplot, yplot, c=np.cos(xplot), cmap=mylist[t],
# #                 edgecolors='none',
# #                 s=np.power(xplot, 4))
# #     plt.pause(1)
# #     plt.cla()
# #     t = (t+1) % len(mylist)

# # # plt.close()

# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # load some test data for demonstration and plot a wireframe
# X, Y, Z = axes3d.get_test_data(0.1)
# ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
