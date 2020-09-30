import os
import requests

import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

hv.extension('bokeh')


data_ranges = [[1970,1979, 'dat'],
               [1960,1969, 'dat'],
               [1950,1959, 'dat'],
               [1940,1946, 'dat'],
               [1930,1939, 'dat'],
               [1920,1929, 'dat'],
               [1910,1919, 'dat'],
               [1900,1909, 'dat'],
               [1888,1899, 'dat']][::-1]

# Download data

def get_decade(start=1920, end=1929, extension='prn'):
  "specify the starting year of the decade eg. 1900, 2010, 2009"
  try:
    link = requests.get(f"https://www.nyse.com/publicdocs/nyse.data/Daily_Share_Volume_{start}-{end}.{extension}")
    file = os.path.join(" ","Data",f"Daily_Share_Volume_{start}-{end}.{extension}")
    
    if link.status_code == 404:
      raise
     else:
      with open(file, 'w') as temp_file:
        temp_file.write(str(link.content.decode("utf-8")))
        
        print2(f"Successfully downloaded {start}-{end}")
  except:
    print2("There was an issue with the download \n\
            You may need a different date range or file extension.\n\
            Check out https://www.nyse.com/data/transactions-statistics-data-library")
    
download_history = [get_decade(decade[0], decade[1], decade[2]) for decade in data_ranges]

# read and format the data
def load_data(start=1920, end=1929, extension="prn"):
  
  # get the path
  path = os.path.join(" ", "Data",f"Daily_Share_Volume_{start}-{end}.{extension}")
  
  if extension = "prn":
    data = pd.read_csv(path, sep='  ', parse_dates=['Date'], engine='python').iloc[2:, 0:2]
    data.loc[:, "  Stock U.S Gov't"] = pd.to_numeric(data.loc[:, "  Stock U.S Gov't"], errors='coerce')
    data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
    data.columns = ['Date', 'Volume']
    return data
  
  else:
    data = pd.read_csv(path)
    data.iloc[:,0] = data.iloc[:,0].apply(lambda x: str(x).strip(' '))
    data = data.iloc[:,0].str.split(' ', 1, expand=True)
    data.columns = ['Date', 'Volume']
    data.loc[:, "Volume"] = pd.to_numeric(data.loc[:, "Volume"], errors='coerce')
    data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
    return data
    
    
data = pd.concat([load_data(decade[0], decade[1], decade[2]) for decade in data_ranges], axis=0)

            
        
  
    
               
