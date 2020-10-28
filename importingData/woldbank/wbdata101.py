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



import wbdata as wb                                                              

wb.get_source() 
wb.get_indicator(source=16)  
