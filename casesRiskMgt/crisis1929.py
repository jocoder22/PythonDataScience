import os
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

hv.extension('bokeh')

# path33 = f"D:\PythonDataScience\casesRiskMgt"
# path33 = f"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\CourseMaterials\Module1\Case Studies in Risk Management Module 1 extra documents"

path33 = f"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\CourseMaterials\Module1"

data_ranges = [[1970,1979, 'dat'],
               [1960,1969, 'dat'],
               [1950,1959, 'dat'],
               [1940,1949, 'dat'],
               [1930,1939, 'dat'],
               [1920,1929, 'prn'],
               [1900,1919, 'dat'],
               [1888,1899, 'dat']][::-1]

"""
# Download data
def get_decade(start=1920, end=1929, extension='prn'):
  "specify the starting year of the decade eg. 1900, 2010, 2009"
  try:
      link = requests.get(f'https://www.nyse.com/publicdocs/nyse/data/Daily_Share_Volume_{start}-{end}.{extension}')
      file = os.path.join(path33,f"Daily_Share_Volume_{start}-{end}.{extension}")
      # print2(link.status_code)
      # print2(link.content.decode("utf-8"))
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
"""

path33 = f"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\CourseMaterials\Module1\data_docs"


# read and format the data
def load_data(start=1920, end=1929, extension="prn"):
  # get the path
  path = os.path.join(path33, "Data",f"Daily_Share_Volume_{start}-{end}.{extension}")
  # path = os.path.join(path33, f"Daily_Share_Volume_{start}-{end}.{extension}")
  
  if extension == "prn":
    data = pd.read_csv(path, sep='   ', parse_dates=['Date'], engine='python').iloc[2:,0:2]
    print2(data.head(), data.columns)
    data.loc[:, "  Stock U.S Gov't"] = pd.to_numeric(data.loc[:, "  Stock U.S Gov't"], errors='coerce')
    data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
    data.columns = ['Date', 'Volume']
    print2(f"Successfully downloaded {start}-{end}")
    return data
  
  else:
    data = pd.read_csv(path)
    data.iloc[:,0] = data.iloc[:,0].apply(lambda x: str(x).strip(' '))
    data = data.iloc[:,0].str.split(' ', 1, expand=True)
    data.columns = ['Date', 'Volume']
    data.loc[:, "Volume"] = pd.to_numeric(data.loc[:, "Volume"], errors='coerce')
    data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
    print2(f"Successfully downloaded {start}-{end}")
    return data



# load data from folder
data = pd.concat([load_data(decade[0], decade[1], decade[2]) for decade in data_ranges], axis=0)

# create plotting object
plot_data = hv.Dataset(data, kdims=['Date'], vdims=['Volume'])

# create scatter plot
black_tuesday = pd.to_datetime('1929-10-29')

vline = hv.VLine(black_tuesday).options(color='#FF7E47')

m = hv.Scatter(plot_data).options(width=700, height=400).redim('NYSE Share Trading Volume').hist()*vline*\
  hv.Text(black_tuesday+pd.DateOffset(months=10), 4e7, "Black Tuesday", halign='left').options(color="#FF7E47")


# plot the daily traded volume against time
plt.scatter(data['Date'], data['Volume'], s=0.1)
plt.axvline(black_tuesday, color="red")
plt.show()


# Download data, not saving here!
def getload_decade(start=1920, end=1929, extension='prn'):
    "specify the starting year of the decade eg. 1900, 2010, 2009"

    webaddress = f'https://www.nyse.com/publicdocs/nyse/data/Daily_Share_Volume_{start}-{end}.{extension}'

    try:
        link = requests.get(webaddress)

        print2(link.status_code)
        if link.status_code == 404:
            raise
            
        else:  
            if extension == "prn":
                data = pd.read_csv(webaddress, sep='   ', parse_dates=['Date'], engine='python').iloc[2:, 0:2]
                print2(data.head(), data.columns)
                data.loc[:, "  Stock U.S Gov't"] = pd.to_numeric(data.loc[:, "  Stock U.S Gov't"], errors='coerce')
                data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
                data.columns = ['Date', 'Volume']
                print2(f"Successfully downloaded {start}-{end}")
                return data
              
            else:
                data = pd.read_csv(webaddress)
                data.iloc[:,0] = data.iloc[:,0].apply(lambda x: str(x).strip(' '))
                data = data.iloc[:,0].str.split(' ', 1, expand=True)
                data.columns = ['Date', 'Volume']
                data.loc[:, "Volume"] = pd.to_numeric(data.loc[:, "Volume"], errors='coerce')
                data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors="coerce")
                print2(f"Successfully downloaded {start}-{end}")
                return data

    except:
        print2("There was an issue with the download \n\
            You may need a different date range or file extension.\n\
            Check out https://www.nyse.com/data/transactions-statistics-data-library")


# download data, form dataframe      
data2 = pd.concat([getload_decade(decade[0], decade[1], decade[2]) for decade in data_ranges], axis=0)
data3 = data2.set_index("Date")

# plt the volume of trade against time
plt.scatter(data2['Date'], data2['Volume'], s=0.1)
plt.axvline(black_tuesday, color="red")
plt.show()


# plot the same data
data2.plot(x="Date", y='Volume', kind="scatter", s=0.1)
plt.axvline(black_tuesday, color="red")
plt.show()

# plot the same data
data3.plot()
plt.axvline(black_tuesday, color="red")
plt.show()