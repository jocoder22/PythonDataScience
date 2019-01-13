# Perform necessary imports
import numpy as np


from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models import Slider, Button
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show
from numpy.random import random

# # Create a new plot: plot
# plot = figure()

# # Add a line to the plot
# plot.line([1,2,3,4,5], [2,5,4,6,7])

# # Add the plot to the current document
# curdoc().add_root(plot)

# from bokeh.io import curdoc
# from bokeh.plotting import figure


# # Create a slider: slider
# slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# # Create a widgetbox layout: layout
# layout = widgetbox(slider)

# # Add the layout to the current document
# curdoc().add_root(layout)




# # Create first slider: slider1
# slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)

# # Create second slider: slider2
# slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

# # Add slider1 and slider2 to a widgetbox
# layout = widgetbox(slider1, slider2)

# # Add the layout to the current document
# curdoc().add_root(layout)



# Create ColumnDataSource: source
# n = 5
# x = np.linspace(0, n, 100)
# y = np.sin(x)
# source = ColumnDataSource(data={'x': x, 'y': y})

# plot = figure()
# # Add a line to the plot
# plot.line('x', 'y', source=source)

# # Create a slider: slider
# slider = Slider(title='my slider', start=0, end=10, step=1, value=n)


# # Define a callback function: callback
# def callback(attr, old, new):

#     # Read the current value of the slider: scale
#     scale = slider.value

#     # Compute the updated y using np.sin(scale/x): new_y
#     new_y = np.sin(scale)

#     # Update source with the new data values
#     source.data = {'x': x, 'y': new_y}

# # Attach the callback to the 'value' property of slider
# slider.on_change('value', callback)

# # Create a column layout: layout
# layout = column(widgetbox(slider), plot)

# output_file('slider.html')
# show(layout)

# Add the layout to the current document
# curdoc().add_root(layout)


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