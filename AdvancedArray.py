import numpy as np

adlist = [[['January', 'February', 'March'],
          ['April', 'May', 'June'],
          ['July', 'August', 'September'],
          ['October', 'November', 'December']],
          [['School', 'Cat', 'Market'],
           ['Pig', 'Dog', 'Cat'],
           ['Rain', 'Wind', 'Fire'],
           ['Books', 'cat', 'Papers']]]

ad = np.array(adlist)


# subsetting np array
adsub = ad[ad != 'Cat']

adindex = ad != "Cat"
adindex2 = np.char.upper(ad) != "Cat".upper()
print(adindex)
print(adindex2)

adsub2 = ad[adindex]
adsub3 = ad[adindex2]


# Predefined indexing
id1 = np.array([[0, 1],
               [0, 1]])

id2 = np.array([[0, 2],
               [1, 1]])

print(ad[id1, id2])

"""
[[['January' 'February' 'March']
 ['Rain' 'Wind' 'Fire']]

 [['April' 'May' 'June']
  ['Pig' 'Dog' 'Cat']]]

"""

id3 = np.array([[0, 2],
               [1, 1]])

print(ad[id1, id2, id3])

"""
[['January' 'Fire']
 ['May' 'Dog']]

 """
