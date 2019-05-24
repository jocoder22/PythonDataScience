#!/usr/bin/env python
import doctest

import requests
from itertools import permutations

url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words.txt'

respond = requests.get(url)

words = respond.text

print(type(words))

mylistword = words.split('\n')
print(mylistword[:11])

print('2' in mylistword)


def formWord(word):
    permutlist = []
    for s in permutations(word):
        new = ''.join(s)
        if new in mylistword:
            permutlist.append(new)

    print(permutlist)

formWord('sicd')

formWord('olm')
