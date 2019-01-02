#!/usr/bin/env python
import sys
# sys.path.insert(0, '/Users/Jose/Desktop/ppp/')
sys.path.append('C:/Users/Jose/Desktop/ppp')

# ### importin file in another folder
# using open to read file 
# path = 'C:/Users/Jose/Desktop/ppp'
# file = open(''.join(path,'timer_function'))

# ## using os
# import os
# os.chdir(path)
# from timer_function import program_performance

from timer_function import program_performance
from scipy.ndimage.filters import gaussian_filter as gf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
import itertools

# 'C:\Users\Jose\Desktop\ppp'
# 'C:/Users/Jose/Desktop/ppp/timer_function'


rand2 = np.random.normal(size=(512, 512))
rand2f = gf(rand2, sigma=10)



style_list = ['seaborn-dark', 'dark_background', 'seaborn-pastel', 'seaborn-colorblind',
    'tableau-colorblind10', 'seaborn-notebook', 'seaborn-dark-palette',
    'grayscale', 'seaborn-poster', 'seaborn', 'bmh', 'seaborn-talk',
    'seaborn-ticks', '_classic_test', 'ggplot', 'seaborn-white', 'classic',
    'Solarize_Light2', 'seaborn-paper', 'fast', 'fivethirtyeight',
    'seaborn-muted', 'seaborn-whitegrid', 'seaborn-darkgrid', 'seaborn-bright',
    'seaborn-deep']

mylist = ['Accent', 'Accent_r', 'Blues', 'CMRmap', 'CMRmap_r', 'Dark2']


""", 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
          'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
          'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
          'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired',        
          'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
          'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
          'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
          'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
          'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
          'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
          'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',
          'YlOrRd', 'YlOrRd_r']"""


xx = np.linspace(0, 2 * np.pi)
xx1 = np.linspace(0, 5, 150)


def f1():
    k = 0
    for i in style_list:
        plt.style.use([i])
        for u in mylist:
            plt.plot(np.sin(xx1), 'r-o')
            plt.title('This is for style: {} and colormap: {}'.format(i, u))
            plt.scatter(xx, np.sin(xx), c=np.cos(xx), cmap=u,
                            edgecolors='none',
                            s=np.power(xx, 5))
            k += 1
        plt.pause(1)
        plt.clf()
    print('k equal {}'.format(k))
    plt.close()


for i in style_list:
    with plt.style.context((i)):
        plt.title('This is for style: {}'.format(i))
        ax1 = plt.subplot2grid((1,2), (0, 0))
        ax1.imshow(rand2f)
        ax2 = plt.subplot2grid((1,2), (0, 1))
        ax2.plot(np.sin(np.linspace(0, 10, 100)), 'r-o')
        plt.pause(1)
        plt.clf()
    
plt.close()




def f2():
    m = 0
    for s, c in itertools.product(style_list, mylist):
        plt.style.use([s])
        plt.plot(np.sin(xx), 'r-o')
        plt.title('This is for style: {} and colormap: {}'.format(s, c))
        plt.scatter(xx, np.sin(xx), c=np.cos(xx), cmap=c,
                            edgecolors='none',
                            s=np.power(xx, 5))
        m += 1
        plt.pause(1)
        plt.clf()
    print('m equal {}'.format(m))
    plt.close()



if __name__ == '__main__':
    program_performance(f1, f2)
        





# text

plt.text(5,0, 'sine curve')
ha ='center'
va = 
size=18 large x-large xx-large
family='monospace' 
weight = bold normal
style =italic
withdash=True
dashlength =24
rotation =45
'sine\ncurve'

plt.gca().yaxis.set_visible(False)
plt.gca().spines['Left'].set_visible(False)
