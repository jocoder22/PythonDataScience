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


# Reverse finding
cordde = (0, 1, 1)
print((id1[cordde], id2[cordde], id3[cordde]))

# Concatenating arrays
cat1 = np.array([[['January', 'Fire'],
                ['May', 'Dog']]], dtype='<U9')

cat2 = np.array([[['January', 'Fire'],
                 ['May', 'Dog']],

                 [['ios', 'kad'],
                 ['oek', 'msn']]], dtype='<U9')

cat22 = np.array([[["ios", "kad"], ["oek", "msn"]]])

cat1.shape  # (1, 2, 2) , the dimensions must march
cat2.shape  # (2, 2, 2) , the dimensions must march
cat22.shape  # (1, 2, 2), again the dimensions must march
cat1220 = np.concatenate((cat1, cat22), axis=0)  # add a new table
"""
[[['January' 'Fire']
  ['May' 'Dog']]

 [['ios' 'kad']
  ['oek' 'msn']]]

"""
cat1221 = np.concatenate((cat1, cat22), axis=1)  # add a new row
"""
[[['January' 'Fire']
  ['May' 'Dog']
  ['ios' 'kad']
  ['oek' 'msn']]]
"""
cat1222 = np.concatenate((cat1, cat22), axis=2)  # add a new column
"""
[[['January' 'Fire' 'ios' 'kad']
  ['May' 'Dog' 'oek' 'msn']]]
"""
