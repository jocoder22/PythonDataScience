import pandas as pd
from urllib.request import urlopen
import csv

site = 'http://mldata.org/repository/data/download/csv/book-evaluation-complete/'
datasiteOpen = urlopen(site)
dataset = pd.read_csv(datasiteOpen, header=None,
                      names=['int0', 'int1', 'int2', 'int3',
                             'int4', 'int5', 'int6'])
dataset.head()


# Loading in chunks: loaded dataset at mydata.csv
bookChunk10 = pd.read_csv(u'~/Desktop/mydata.csv', header=None,
                          names=['int0', 'int1', 'int2', 'int3',
                                 'int4', 'int5', 'int6'],
                          chunksize=10)

for chunk in bookChunk10:
    print('Shape:', chunk.shape)
    print(chunk, '\n')

# Loading as iterator
bookIter = pd.read_csv(u'~/Desktop/mydata.csv', header=None,
                       names=['int0', 'int1', 'int2', 'int3',
                              'int4', 'int5', 'int6'],
                       iterator=True)

print(bookIter.get_chunk(10).shape)
next20 = bookIter.get_chunk(20)
next20


# Reading using csv.DictReader
with open(r'C:/Users/Jose/Desktop/book.csv', 'rt') as stream:
    mydict = dict()
    mylist = list()
    for n, row in enumerate(csv.DictReader(stream,
                                           fieldnames=['int0', 'int1', 'int2',
                                                       'int3', 'int4', 'int5',
                                                       'int6'],
                                           dialect='excel')):
        for (k, v) in row.items():
            mydict[k] = v
            mylist.append(mydict)
    # below using dictionary comprehension
        # mylist4 = {k:v for (k,v) in row.items()}
        # mylist44.append(mydict)
    # mylist2 = [{k:v} for n, row in enumerate(forma) for (k,v) in row.items()]

mylist[:3]

with open(r'C:/Users/Jose/Desktop/book.csv', 'rt') as stream:
    mydict = dict()
    mylist = list()
    for n, row in enumerate(csv.DictReader(stream,
                                           fieldnames=['int0', 'int1', 'int2',
                                                       'int3', 'int4', 'int5',
                                                       'int6'],
                                           dialect='excel')):
        if n <= 3:
            print(n, row)
        else:
            break


# Using csv.reader
with open(r'C:/Users/Jose/Desktop/book.csv', 'rt') as dstream:
    for n, row in enumerate(csv.reader(dstream, dialect='excel')):
            if n <= 3:
                print(n, row)
            else:
                break
