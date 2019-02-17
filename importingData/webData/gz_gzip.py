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

r = requests.get(url_1)

rfile = io.BytesIO(r.content)
tts = []

with gzip.GzipFile(fileobj=rfile) as gzfile:
    for chunk in pd.read_csv(gzfile, chunksize=100000):
        tts.append(chunk)

df = pd.concat(tts)

print(df.shape, df.columns, sep='\n')

print(sys.getsizeof(r), sys.getsizeof(rfile), sys.getsizeof(gzfile), sys.getsizeof(df), sep='\n')

del r; del rfile; del gzfile; del df