#!/usr/bin/env python
import sys
import io
import requests
import gzip
import numpy as np
import pandas as pd
import os

path = "C:\\Users\\Jose\\Desktop\\TimerSeriesAnalysis\\police_nj\\"

os.chdir(path)

url_1 = 'https://stacks.stanford.edu/file/druid:py883nd2578/NJ-clean.csv.gz'
url = 'https://stacks.stanford.edu/file/druid:py883nd2578/WY-clean.csv.gz'

r = requests.get(url_1, stream=True)

rfile = io.BytesIO(r.content); del r
tts = []

with gzip.GzipFile(fileobj=rfile) as gzfile:
    for chunk in pd.read_csv(gzfile, chunksize=100000, low_memory=False):
        tts.append(chunk)

df = pd.concat(tts)

print(df.shape, df.columns, sep='\n')

print(sys.getsizeof(tts), sys.getsizeof(rfile), sys.getsizeof(gzfile), sys.getsizeof(df), sep='\n')

del rfile; del gzfile; 
del df; del tts



####################### Method Two
# url = 'https://stacks.stanford.edu/file/druid:py883nd2578/WY-clean.csv.gz'

# # Download, read, and form dataframe
# filename = url.split('/')[-1]
# with open(filename, "wb") as f:
#     r = requests.get(url)
#     f.write(r.content)
#     with gzip.open(filename, 'rb') as gzfile:
#         wy = pd.read_csv(gzfile)
#         wy.to_csv('Data-wy.csv', index=False)

# try:
#     os.remove(filename)
# except OSError:
#     pass
    