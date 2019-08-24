#!/usr/bin/env python
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import folium


path = r'C:\Users\Jose\Desktop\PythonDataScience\geospatialAnalysis'
os.chdir(path)
sp = '\n\n'


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
worldcity = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))


# Select the country Nigeria from the map
nigeria = world[world['name'] == 'Nigeria']

nigeria.plot()
plt.show()

usa = world.loc[world['iso_a3'] == 'USA', 'geometry'].squeeze()

# cities in Nigeria
nigeriaCities = worldcity[worldcity.within(nigeria)]
print(nigeriaCities, sep=sp, end=sp)
