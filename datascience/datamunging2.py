import pandas as pd 
from urllib.request import urlopen

site = 'http://mldata.org/repository/data/download/csv/book-evaluation-complete/'
datasiteOpen = urlopen(site)
dataset = pd.read_csv(datasiteOpen, header=None, names=['int0','int1','int2','int3','int4','int5','int6'])
dataset.head()


# Loading in chunks: loaded dataset at mydata.csv
bookChunk10 = pd.read_csv(u'~/Desktop/mydata.csv', header=None, 
                          names=['int0','int1','int2','int3','int4','int5','int6'],
                          chunksize=10) 

for chunk in bookChunk10: 
    print ('Shape:', chunk.shape) 
    print (chunk,'\n')

# Loading as iterator
bookIter = pd.read_csv(u'~/Desktop/mydata.csv', header=None, 
                       names=['int0','int1','int2','int3','int4','int5','int6'],
                       iterator=True)

print(bookIter.get_chunk(10).shape)
next20 = bookIter.get_chunk(20)  
next20