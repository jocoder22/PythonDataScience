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


world.loc[world.continent=='Africa'].plot()
# plt.axis("off")
plt.xticks([]),
plt.yticks([])
plt.show()


world.loc[world.continent.str.contains('America')].plot()
plt.tick_params(
    axis='both',                    # changes apply to the both-axis, may be x or y
    which='both',                   # both major and minor ticks are affected
    bottom=False,                   # ticks along the bottom edge are off
    left=False, right= False,      
    top=False,                      # ticks along the top edge are off
    labelbottom=False,
    labeltop=False, labelleft=False, labelright=False)
plt.show()