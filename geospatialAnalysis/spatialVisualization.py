import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


print(gpd.datasets.available)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print(world.head())


world.plot()
# plt.axis("off")
plt.xticks([]),
plt.yticks([])
plt.show()


