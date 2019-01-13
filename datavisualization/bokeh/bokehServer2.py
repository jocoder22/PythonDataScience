# Perform necessary imports
import numpy as np


from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models import Slider
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show
from numpy.random import random
import random



# Create ColumnDataSource: source
n = 5
x = np.linspace(0, n, 100)
y = np.sin(x)
# source = ColumnDataSource(data={'x': np.linspace(0, n, 100), 'y': np.sin(np.linspace(0, n, 100))})
source = ColumnDataSource(data={'x': x, 'y': y})

plot = figure()


# Create a slider: slider
slider = Slider(title='my slider', start=0, end=20, step=1, value=5)


# Define a callback function: callback
def callback(attr, old, new):
    global n
    # Read the current value of the slider: scale
    n = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    
    x = np.linspace(0, n, 100)
    y = np.sin(np.linspace(0, n, 100)) + np.sin(2 + x)
    # Update source with the new data values
    source.data = {'x': x, 'y': y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# output_file('slider.html')
# show(layout)
# layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)




# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)


# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
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
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)



# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data={'x': x, 'y': y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)


# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models

from bokeh.models import CheckboxGroup, RadioGroup, Toggle
# Add a Toggle: toggle
toggle = Toggle(button_type='success' , label='Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))
