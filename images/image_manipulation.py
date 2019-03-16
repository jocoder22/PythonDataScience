#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms


path = r"C:\Users\Jose\Pictures"
os.chdir(path)

img = plt.imread('pic11.png')
plt.axis('off')
plt.imshow(img)
plt.show()

img.shape

# # show the three channels
# for i in range(3):
#     onechannel = img[..., i]
#     plt.imshow(onechannel)
#     plt.axis('off')
#     plt.pause(1)
#     plt.clf()

# plt.close()

# reverse the channels
rev_color = img[..., ::-1]
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.imshow(rev_color)
plt.axis('off')
plt.show()

# turn image upside down
updown = img[::-1]
invert_mirror = img[::-1,::-1,:]
mirror_image = img[:,::-1,:]


plt.subplot(141)
plt.imshow(img)
plt.title('original')
plt.axis('off')

plt.subplot(142)
plt.imshow(updown)
plt.axis('off')
plt.title('upsidedown')

plt.subplot(143)
plt.imshow(invert_mirror)
plt.axis('off')
plt.title('invert_mirror')

plt.subplot(144)
plt.imshow(mirror_image)
plt.axis('off')
plt.title('mirror_image')
plt.show()


# for i in range(0,360, 10):
#     ax = plt.subplot(111)
#     tr = transforms.Affine2D().rotate_deg(i)
#     plt.imshow(img, transform=tr + ax.transData)
#     plt.axis('off')
#     plt.pause(1)
#     plt.clf()

# fig = plt.figure()
# ax = fig.add_subplot(111)

# tr = transforms.Affine2D().rotate_deg(30)
# ax.imshow(img, transform=tr+ ax.transData)
# plt.show()


def do_plot(ax, Z, t):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transforms.Affine2D().rotate_deg(t) + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    # ax.set_xlim(-0.5, 599.5)
    # ax.set_ylim(-0.5, 799.5)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-4, 4)
   

# fig, ax1, = plt.subplot(111)
for i in range(0,360,10):
    ax1 = plt.subplot(111)
    im = plt.imshow(img, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transforms.Affine2D().rotate_deg(i) + ax1.transData
    im.set_transform(trans_data)

    # # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax1.plot([x1, x2, x2, x1, x1,x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1,y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    plt.axis('off')
    plt.pause(1)
    plt.clf()