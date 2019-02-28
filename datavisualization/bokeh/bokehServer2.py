# Perform necessary imports
import os
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models import Slider, Button
from bokeh.models import CheckboxGroup, RadioGroup, Toggle
from bokeh.models import ColumnDataSource, Select
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show
from numpy.random import random
import random

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\datavisualization\\bokeh\\outfiles\\'
os.chdir(path)


# create figure
p1 = figure()

# Create a slider: slider
slider = Slider(title='my slider', start=10, end=20, step=1, value=20)

n = slider.value
xx = np.linspace(0, n, 100)
yy = np.sin(xx)

# Create ColumnDataSource: source
# source = ColumnDataSource(data={'x': np.linspace(0, n, 100), 'y': np.sin(np.linspace(0, n, 100))})
source = ColumnDataSource(data={'x': xx, 'y': yy})

# Add a line to the plot
p1.line('x', 'y', source=source)

# Define a callback function: callback
def callback(attr, old, new):
    global n
    # Read the current value of the slider: scale
    n = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    
    # x = np.linspace(0, n, 100)
    yy = np.sin(np.linspace(0, n, 100)) + np.sin(2 + xx)
    # Update source with the new data values
    source.data = {'x': xx, 'y': yy}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)



# Create a column layout: layout
col = column(widgetbox(slider), p1)


# output_file('slider.html')
# show(layout)
# layout = widgetbox(slider)

# Add the layout to the current document
# curdoc().add_root(col)




# Access the European dataset
url = 'https://www.eea.europa.eu/data-and-maps/figures/correlation-between-fertility-and-female-education/trend01-5g-soer2010-xls/at_download/file'

# download named spread sheet
xl = pd.read_excel(url,sheet_name='data COMPILATION',skiprows=7,nrows=162)

fertility = xl.fertility
female_literacy= xl['female literacy']
population = xl.population

# Create ColumnDataSource: source
source2 = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
p2 = figure()

# Add circles to the plot
p2.circle('x', 'y', source=source2)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source2.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source2.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select2 = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select2.on_change('value', update_plot)



layout2 = column(select2, p2)
layout4 = row(col, layout2)
curdoc().add_root(layout4)



####################################
p3 = figure()
# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback3(attr, old, new):
    # If select1 is 'A' 
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback3)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)



# Create a Button with label 'Update Data'
button = Button(label='Update Data')
N = 200
x1 = np.linspace(0, 10, N)
y1 = np.sin(x1)
source5 = ColumnDataSource(data={'x': x1, 'y': y1})
# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y1 = np.sin(x1 + np.random.random(N))

    # Update the ColumnDataSource data dictionary
    source5.data={'x': x1, 'y': y1}

# Add the update callback to the button
button.on_click(update)

p3.circle('x', 'y', source=source5)

# Create layout and add to current document
layout = column(widgetbox(button), p3)
curdoc().add_root(layout)


# Add a Toggle: toggle
toggle = Toggle(button_type='success' , label='Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))


# # # bokeh serve --show bokehServer2.py
