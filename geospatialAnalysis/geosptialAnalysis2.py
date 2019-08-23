#!/usr/bin/env python
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import folium
from IPython.display import HTML

path = r'C:\Users\Jose\Desktop\PythonDataScience\geospatialAnalysis'
os.chdir(path)
sp = '\n\n'

print(gpd.datasets.available)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print(world.head(), world.shape, world.columns, sep=sp, end=sp)
print(world.loc[:,['name', 'iso_a3']].head())


worldcity = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
print(worldcity.head(), worldcity.shape, worldcity.columns, sep=sp, end=sp)

newyork = gpd.read_file(gpd.datasets.get_path('nybb'))
print(newyork.head(), newyork.shape, newyork.columns, sep=sp, end=sp)


# plot world cities on the world map
fig, ax = plt.subplots(figsize=(8, 26), dpi=80)
world.plot(ax=ax)
worldcity.plot(ax=ax, color='red', markersize=1)
ax.set_axis_off()
plt.show()


# using folium map
VaticanCity = worldcity.geometry[0]
VaticanCityR = [VaticanCity.y, VaticanCity.x] 
print(VaticanCity)

# Construct a folium map with Vatican City
downtown_map = folium.Map(location= VaticanCityR, zoom_start = 15)

# Save the HTML map
downtown_map.create_map(path='VaticanCity.html')

