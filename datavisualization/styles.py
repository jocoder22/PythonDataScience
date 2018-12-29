#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter as gf
import matplotlib.pyplot as plt
import numpy as np


rand2 = np.random.normal(size=(512, 512))
rand2f = gf(rand2, sigma=10)

style_list = ['seaborn-dark', 'dark_background', 'seaborn-pastel', 'seaborn-colorblind',
 'tableau-colorblind10', 'seaborn-notebook', 'seaborn-dark-palette',
 'grayscale', 'seaborn-poster', 'seaborn', 'bmh', 'seaborn-talk',
 'seaborn-ticks', '_classic_test', 'ggplot', 'seaborn-white', 'classic',
 'Solarize_Light2', 'seaborn-paper', 'fast', 'fivethirtyeight',
 'seaborn-muted', 'seaborn-whitegrid', 'seaborn-darkgrid', 'seaborn-bright',
 'seaborn-deep']



mylist = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
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
          'YlOrRd', 'YlOrRd_r']

for i in mylist:
    plt.title('This is for style: ' + i)
    # plt.style.use(i)
    # plt.imshow(rand2f)
    plt.imshow(rand2f, cmap=i)
    plt.colorbar()
    plt.pause(1)
    plt.clf()

plt.close()