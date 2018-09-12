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

adsub2 = ad[adindex]
adsub3 = ad[adindex2]
