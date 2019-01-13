
import pandas as pd
# Import figure from bokeh.plotting
from bokeh.plotting import figure
import datetime
import pandas_datareader as pdr

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
          'LAT':'blue', 'NAM':'green', 'OCE':'teal'}


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


# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)




# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='saddlebrown', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='teal', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)



# Load apple dataset from Yahoo finance
symbol = 'AAPL'
starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2013, 12, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
print(apple.columns)
print(apple.index.name)


# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(apple.index, apple['Adj Close'])

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


# Load apple dataset from Yahoo finance
symbol = 'AAPL'
starttime = datetime.datetime(2000, 3, 1)
endtime = datetime.datetime(2000, 7, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
print(apple.columns)
print(apple.index.name)

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(apple.index, apple['Adj Close'])

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(apple.index, apple['Adj Close'], fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('lineCircle.html')
show(p)


# making patches using list of lists for the cordinates
x = [[1,2.2,5,3,4],[1,4,3.2,2],[5,3,4,3.2,5.8,9,7.2],[2.2,5,6.3,5.2],[6.3,7.2,10,8,7.2,5]]
y = [[4,2,3,3.8,5],[4,5,7,6],   [3,3.8,5,7,8,3.2,4.2], [2,3,2,1.6], [2,1,2.8,2.4,4.2,3]]

p = figure(x_axis_label='Lattitude(degrees)', y_axis_label='Longitude(Degrees)')
# Add patches to figure p with line_color=white for x and y
p.patches(x,y, line_color='white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)



# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(xl)

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle glyphs to the figure p
p.circle('fertility','female literacy', color='ccmap', size=10, source=source)

# Specify the name of the output file and show the result
output_file('fertlity.html')
show(p)



# Create a figure with the "box_select" tool: p
source = ColumnDataSource(xl)
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)', 
        tools='box_select,reset,pan,wheel_zoom,lasso_select,ybox_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('fertility','female literacy', selection_color='red',nonselection_alpha=0.1,
        color='ccmap', size=10, source=source)


# Specify the name of the output file and show the result
output_file('fertlity2.html')
show(p)





# import the HoverTool
from bokeh.models import HoverTool
source = ColumnDataSource(apple)

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Daily Volume', y_axis_label='Adjusted Close(US Dollars)')

# Add circle glyphs to figure p
p.circle('Date', 'Adj Close', size=10,
         fill_color='grey', alpha=0.3, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white', source=source)
p.line(apple.index, apple['Adj Close'])
# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p

p.add_tools(hover)
# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)






#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(xl)
# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['AF', 'ASI', 'EUR', 'OCE', 'NAM', 'LAT' ],
                                      palette=['red', 'green', 'blue', 'teal', 'cyan', 'magenta'])

# Add a circle glyph to the figure p
p.circle('fertility','female literacy', source=source,
            color=dict(field='Continent', transform=color_mapper),
            legend='Continent')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)
