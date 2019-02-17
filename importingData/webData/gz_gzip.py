#!/usr/bin/env python
import sys
import io
import requests
import gzip
import numpy as np
import pandas as pd

url = 'https://stacks.stanford.edu/file/druid:py883nd2578/WY-clean.csv.gz'

r = requests.get(url)

rfile = io.BytesIO(r.content)

with gzip.GzipFile(fileobj=rfile) as gzfile:
    df = pd.read_csv(gzfile)

print(df.shape, df.columns, sep='\n')

print(sys.getsizeof(r), sys.getsizeof(rfile), sys.getsizeof(gzfile), sys.getsizeof(df), sep='\n')

