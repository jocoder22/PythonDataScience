#!/usr/bin/env python
import os
import geopandas as gpd
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import folium
from shapely.geometry import Point


path = r'C:\Users\Jose\Desktop\PythonDataScience\geospatialAnalysis'
os.chdir(path)
sp = '\n\n'



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
worldcity = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))


# Select the country Nigeria from the map
nigeria = world[world['name'] == 'Nigeria']

nigeria.plot()
plt.show()


# cities in Nigeria
nigeriaGeo = nigeria['geometry'].squeeze()
nigeriaCities = worldcity[worldcity.within(nigeriaGeo)]
print(nigeriaCities, nigeriaCities.crs, sep=sp, end=sp)


# plot Abuja map
locat = [nigeriaCities.geometry.y, nigeriaCities.geometry.x] 
abujaMap = folium.Map(location=locat, tiles='Stamen Terrain', zoom_start = 10)
abujaMap.add_child(folium.LatLngPopup())

# save the map
abujaMap.save('AbujaMap.html')


# plot Awaka map
loca = [6.2020, 7.0834]
ploc = [2.1653113509801, 4.66518765021497]
awkaMap = folium.Map(location=ploc, zoom_start = 10)

popup = f'Name: ThirdP\nLatitude: {ploc[0]}\nLongitude: {ploc[1]}'
folium.Marker(location=ploc, popup=popup).add_to(awkaMap)
# awkaMap.add_child(folium.LatLngPopup())

# save the map
awkaMap.save('AwkaMap.html')



pt1 = Point(519763.561, 241302.699)
pt2 = Point(519901.095, 241041.357)
pt3 = Point(2.1653113509801, 4.66518765021497)


geometry = list([pt1, pt2])
geometry2 = [pt3]
geo_data = gpd.GeoDataFrame(crs={'init': 'epsg:3857'}) # this is in meters
geo_data2 = gpd.GeoDataFrame(crs={'init': 'epsg:4326'}) # this is in decimal degrees
geo_data.loc[1, 'name'] = 'FirstP'
geo_data.loc[2, 'name'] = 'SecondP'
geo_data2.loc[1, 'name'] = 'ThirdP'

geo_data.geometry = geometry
geo_data2.geometry = geometry2

print(geo_data, geo_data.crs, geo_data2, geo_data2.crs, end=sp, sep=sp)



