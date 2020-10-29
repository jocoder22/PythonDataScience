# https://wbdata.readthedocs.io/en/latest/wbdata_library.html
# https://wbdata.readthedocs.io/en/latest/

# wbdata.search_indicators
# wbdata.search_countries
# wbdata.get_source
# wbdata.get_topic
# wbdata.get_lendingtype - Retrieve information on an income level aggregate
# wbdata.get_incomelevel  - Retrieve information on an income level aggregate
# wbdata.get_country
# wbdata.get_indicator
# wbdata.get_data
# wbdata.get_series
# wbdata.get_dataframe


# wbdata.get_dataframe(indicators, country='all', data_date=None, freq='Y', source=None, convert_date=False, keep_levels=False, cache=True)
# Convenience function to download a set of indicators and merge them into a pandas DataFrame. 
#The index will be the same as if calls were made to get_data separately.

# Indicators
# An dictionary where the keys are desired indicators and the values are the desired column names

# Country
# a country code, sequence of country codes, or “all” (default)

# Data_date
# the desired date as a datetime object or a 2-sequence with start and end dates

# Freq
# the desired periodicity of the data, one of ‘Y’ (yearly), ‘M’ (monthly), or ‘Q’ (quarterly). 
# The indicator may or may not support the specified frequency.

# Source
# the specific source to retrieve data from (defaults on API to 2, World Development Indicators)



# Convert_date
# if True, convert date field to a datetime.datetime object.

# Keep_levels
# if True don’t reduce the number of index levels returned if only getting one date or country

# Cache
# use the cache

# Returns
# a WBDataFrame


import pandas as pd
import wbdata as wb  
import datetime 

# search for data sources in world bank data
wb.get_source() 
wb.get_indicator(source=16)  

# do country search
wb.search_countries('united') 

# do wild search
wb.search_countries('niger*') 

# get data for country
# SE.ADT.1524.LT.FM.ZS  Literacy rate, youth (ages 15-24), gender parity index (GPI)
# return a multi-dictionary(based on year) list
wb.get_data("SE.ADT.1524.LT.FM.ZS", country="USA")


# selecting data range
date_range = datetime.datetime(2008, 1, 1), datetime.datetime(2019, 1, 1)                                                                                             
# SH.CON.1524.FE.ZS     Condom use, population ages 15-24, female (% of females ages 15-24)
# SH.CON.1524.MA.ZS     Condom use, population ages 15-24, male (% of males ages 15-24)
wb.get_data("SH.CON.1524.FE.ZS", country=["USA", "GBR", "NGA"], data_date=date_range)


# search for indicator of interest
wb.search_indicators("gdp per capita") 
wb.search_indicators("condom use") 


# get income level classes
wb.get_incomelevel() 


# let get the data in pandas
countries = [i['id'] for i in wb.get_country(incomelevel='HIC')]                                                                                                 

indicators = {"IC.BUS.EASE.XQ": "doing_business", "NY.GDP.PCAP.PP.KD": "gdppc"}         

df = wb.get_dataframe(indicators, country=countries, convert_date=True, data_date=date_range)   

# do exploratory data analysis
df.groupby('country').describe()
df.groupby("country").describe()['gdppc']['mean'].reset_index().sort_values(by='mean', ascending=False)
df.groupby("country")['gdppc'].mean().reset_index().sort_values(by='gdppc', ascending=False)

df.sort_index().dropna().groupby('country').last().corr()  
