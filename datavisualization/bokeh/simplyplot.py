
import pandas as pd
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

url = 'https://www.eea.europa.eu/data-and-maps/figures/correlation-between-fertility-and-female-education/trend01-5g-soer2010-xls/at_download/file'

# download the whole workbook
xlworkbook = pd.read_excel(url,sheet_name=None)
print(xlworkbook.keys())

# download named spread sheet
xl = pd.read_excel(url,sheet_name='data COMPILATION',skiprows=7,nrows=162)


# Print the column name to the shell
print(xl.keys())
# Index(['Country ', 'Continent', 'female literacy', 'fertility', 'population'], dtype='object')


colors = {'AF':'yellow','ASI':'magenta', 'EUR':'cyan', 
          'LAT':'blue', 'NAM':'green', 'OCE':'red'}


xl['ccmap'] = xl['Continent'].map(colors)
