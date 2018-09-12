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

adsub2 = ad[adindex]
