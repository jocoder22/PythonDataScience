#!/usr/bin/env python
# import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
sp = '\n\n'

print(gpd.datasets.available)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print(world.head(), world.shape, world.columns, sep=sp, end=sp)