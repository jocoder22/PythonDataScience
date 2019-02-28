# Perform necessary imports
import os 
import numpy as np


from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models import Slider, Button
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show
from numpy.random import random

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\datavisualization\\bokeh\\outfiles\\'
os.chdir(path)

N = 300
source = ColumnDataSource(data={'x': random(N), 'y': random(N)})

plot = figure()
# Create a slider: slider
slider = Slider(title='my slider', start=100, end=1000, step=10, value=N)

plot.circle('x', 'y', source=source)
# Define a callback function: callback
def callback2(attr, old, new):

    # Read the current value of the slider: scale
    N = slider.value

    # Update source with the new data values
    source.data = {'x': random(N), 'y': random(N)}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback2)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# output_file('slider.html')
# show(layout)
# Add the layout to the current document
curdoc().add_root(layout)

# cd to the folder and Run
# bokeh serve --show bokehServer.py
