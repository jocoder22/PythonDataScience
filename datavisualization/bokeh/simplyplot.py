
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


fertility = xl['fertility']
female_literacy = xl['female literacy']

latinos = xl['Continent'] == 'LAT'
fertility_latinamerica = xl.fertility[latinos]
female_literacy_latinamerica = xl['female literacy'][latinos]

africans = xl['Continent'] == 'AF'
fertility_africa = xl.fertility[africans]
female_literacy_africa = xl['female literacy'][africans]

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot

show(p)


# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica,female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa,female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)
