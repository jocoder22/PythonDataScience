import pandas as pd
import datetime
import pandas_datareader as pdr

# Import figure from bokeh.plotting
from bokeh.plotting import figure
from bokeh.plotting import ColumnDataSource

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show
# Import row from bokeh.layouts
from bokeh.layouts import row, column, gridplot

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs

url = 'https://www.eea.europa.eu/data-and-maps/figures/correlation-between-fertility-and-female-education/trend01-5g-soer2010-xls/at_download/file'

# download named spread sheet
xl = pd.read_excel(url,sheet_name='data COMPILATION',skiprows=7,nrows=162)


# Print the column name to the shell
print(xl.keys())


colors = {'AF':'yellow','ASI':'magenta', 'EUR':'cyan', 
          'LAT':'blue', 'NAM':'green', 'OCE':'teal'}

fullcontinent = {'AF':'Africa','ASI':'Asia', 'EUR':'Europe', 
          'LAT':'Latin America', 'NAM':'North America', 'OCE':'Oceanic'}

xl['ccmap'] = xl['Continent'].map(colors)
xl['Fcontinent'] = xl.Continent.map(fullcontinent)
xl['country'] = xl['Country ']

print(xl.head())


# Load apple dataset from Yahoo finance
symbol = 'AAPL'
starttime = datetime.datetime(2003, 1, 1)
endtime = datetime.datetime(2018, 12, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
print(apple.columns)
print(apple.index.name)

# # Create a ColumnDataSource from xl: source
# source = ColumnDataSource(xl)

# # Create the first figure: p1
# p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')

# # Add a circle glyph to p1
# p1.circle('fertility', 'female literacy', source=source)

# # Create the second figure: p2
# p2 = figure(x_axis_label='population', y_axis_label='female literacy (% population)')

# # Add a circle glyph to p2
# p2.circle('population', 'female literacy', source=source)

# # Put p1 and p2 into a horizontal row: layout
# layout = row(p1, p2)

# # Specify the name of the output_file and show the result
# output_file('fert_row.html')
# show(layout)



# ######## column layout
# # Create a ColumnDataSource from xl: source
# source = ColumnDataSource(xl)
# # Create a blank figure: p1
# p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')

# # Add circle scatter to the figure p1
# p1.circle('fertility', 'female literacy', source=source)

# # Create a new blank figure: p2
# p2 = figure(x_axis_label='population', y_axis_label='female literacy (% population)')

# # Add circle scatter to the figure p2
# p2.circle('population', 'female literacy', source=source)

# # Put plots p1 and p2 in a column: layout
# layout = column(p1, p2)

# # Specify the name of the output_file and show the result
# output_file('fert_column.html')
# show(layout)

################################
# Using gridplot, same as multiple rows, and columns
# Create a list containing plots p1 and p2: row1
# a2002 = apple.loc['2003':'2006',]
# sp1 = ColumnDataSource(a2002)
# a2007 = ColumnDataSource(apple.loc['2007':'2010',])
# a2011 = ColumnDataSource(apple.loc['2011':'2014',])
# a2015 = ColumnDataSource(apple.loc['2015':'2018',])


# # Create a figure with x_axis_type="datetime": p
# p1 = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# p2 = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# p3 = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# p4 = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# # Plot date along the x axis and price along the y axis
# p1.line('Date', 'Adj Close', source=sp1)
# p2.line('Date', 'Adj Close', source=a2007 )
# p3.line('Date', 'Adj Close', source=a2011)
# p4.line('Date', 'Adj Close', source=a2015)

# # Create a list containing plots p1 and p2: row1
# row1 = [p1, p2]

# # Create a list containing plots p3 and p4: row2
# row2 = [p3, p4]

# # Create a gridplot using row1 and row2: layout
# layout = gridplot([row1, row2])

# # Specify the name of the output_file and show the result
# output_file('grid.html')
# show(layout)


################## linked brushing
# Create a ColumnDataSource from xl: source
# source = ColumnDataSource(xl)

# # Create the first figure: p1
# p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
#             tools='box_select,lasso_select')

# # Add a circle glyph to p1
# p1.circle('fertility', 'female literacy', source=source)

# # Create the second figure: p2
# p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
#             tools='box_select,lasso_select')

# # Add a circle glyph to p2
# p2.circle('fertility', 'population', source=source)

# # Create row layout of figures p1 and p2: layout
# layout = row([p1, p2])

# # Specify the name of the output_file and show the result
# output_file('linked_brush.html')
# show(layout)


############## create legend
# Create a ColumnDataSource from xl: source
latin_america = ColumnDataSource(xl[xl['Continent'] == 'LAT'])
africa = ColumnDataSource(xl[xl['Continent'] == 'AF'])

p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')

# Add the first circle glyph to the figure p
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)





latin_america = ColumnDataSource(xl[xl['Continent'] == 'LAT'])
africa = ColumnDataSource(xl[xl['Continent'] == 'AF'])
# Assign the legend to the bottom left: p.legend.location
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')

# Add the first circle glyph to the figure p
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female literacy', source=africa, size=10, color='blue', legend='Africa')

p.legend.location = 'bottom_left'
# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'


# Creating hover tools
# Create a HoverTool object: hover
hover = HoverTool(tooltips=[('Country ','@country'),
                            ('Continent','@Fcontinent'),
                            ('Population', ' @population'),
                            ('Fertility', ' @fertility')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)




############## tab

latin_america = ColumnDataSource(xl[xl['Continent'] == 'LAT'])
europe = ColumnDataSource(xl[xl['Continent'] == 'EUR'])
ocenic = ColumnDataSource(xl[xl['Continent'] == 'OCE'])
africa = ColumnDataSource(xl[xl['Continent'] == 'AF'])
asia = ColumnDataSource(xl[xl['Continent'] == 'ASI'])
North_america = ColumnDataSource(xl[xl['Continent'] == 'NAM'])

# Assign the legend to the bottom left: p.legend.location
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')

# Add the first circle glyph to the figure p
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')
p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend='Latin America')


# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)



##### linked ranges
  # Link the x_range of p2 to p1: p2.x_range
p2.x_range =  p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range =  p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range =  p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range =  p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)
