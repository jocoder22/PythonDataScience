import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")

path = r"D:\PythonDataScience\datavisualization\Uda_datavisualization"
os.chdir(path)

fcon = pd.read_csv('fuel-econ.csv')

url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2020-03-13/visualisations/listings.csv"

airbnb = pd.read_csv(url)

print2(airbnb.head(), airbnb.columns, airbnb.info())


"""

# bivariate both numeric
plt.scatter(data=airbnb, x="latitude", y="longitude")
plt.show()

tips = sns.load_dataset("tips")
ax = sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()


ax = sns.regplot(x="city", y="highway", data=fcon, x_jitter=0.03, 
                    scatter_kws={"alpha": 1/20
            })
plt.show()


plt.scatter(data=airbnb, x="price", y="longitude")
plt.show()
"""
sns.regplot(x="displ", y="comb", data=fcon, x_jitter=0.03, scatter_kws={
                    "alpha": 1/20
            })
plt.show()



print2(fcon[['displ','comb']].describe())
bin_x = np.arange(fcon.displ.min(), fcon.displ.max()+0.5, 0.5)
bin_y = np.arange(fcon.comb.min(), fcon.comb.max()+3, 3)
his2d = plt.hist2d(x="displ", y="comb", data=fcon, cmin=0.5,
            cmap='viridis_r', bins=[bin_x, bin_y]
            )
plt.xlabel("Displacement (l)")
plt.ylabel("CO2 (g/ml)")
counts = his2d[0]

# loop through the cell counts and add text annotations for each
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        c = counts[i,j]
        if c >= 20: # increase visibility on darkest cells
            plt.text(bin_x[i]+0.25, bin_y[j]+1.5, int(c),
                     ha = 'center', va = 'center', color = 'white')
        elif c > 0:
            plt.text(bin_x[i]+0.25, bin_y[j]+1.5, int(c),
                     ha = 'center', va = 'center', color = 'black')
plt.show()

"""
sns.regplot(x="year", y="comb", data=fcon, x_jitter=0.34, scatter_kws={
                    "alpha": 1/20
            })
plt.show()


# airbnb2 = airbnb.loc[airbnb.price < 200]
# airbnb2['logprice'] = np.log10(airbnb2['price'])
# sns.regplot(data=airbnb2, x="logprice", y="number_of_reviews")
# plt.show()

"""
# one quantitative and qualitative variable
vclass = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars',  'Large Cars']
cat_ordered = pd.api.types.CategoricalDtype(ordered = True, categories = vclass)
fcon['VClass'] = fcon['VClass'].astype(cat_ordered)
ax = sns.violinplot(x="VClass", y="comb", data=fcon,
        color=sns.color_palette()[0], inner=None)
plt.xticks(rotation=15)
plt.show()



ax = sns.violinplot(x="comb", y="VClass", data=fcon,
        color=sns.color_palette()[0], inner="quartile")
# plt.xticks(rotation=15)
plt.show()


ax = sns.boxplot(x="VClass", y="comb", data=fcon,
        color=sns.color_palette()[0])
plt.xticks(rotation=15)
plt.show()



# two categorical variables
fcon['trans_type'] = fcon['trans'].apply(lambda x: x.split()[0])
# fig1 = plt.figure(facecolor='white')
# ax1 = plt.axes(frameon=False)
# ax1.set_frame_on(False)
# Hide the right and top spines
ax1 = plt.subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
# ax1.axes.get_yaxis().set_visible(False)
# ax1.axes.get_xaxis().set_visible(False)
ax1 = sns.countplot(x="VClass", hue="trans_type", data=fcon)
# plt.xticks(rotation=15)
ax1.legend(loc = 8, ncol = 3, framealpha = 1, title = 'Transmission Type')
plt.show()


ax1 = plt.subplot(111)
cat_count = fcon.groupby(["VClass", "trans_type"]).size()
cat_count = cat_count.reset_index(name = "Counts")
cat_count = cat_count.pivot(index = 'VClass', columns = 'trans_type', values="Counts")
ax1 = sns.heatmap(cat_count, annot=True, fmt='d')
plt.show()


# facetting, categorical and numeric variables
group_means = fcon.groupby(['VClass']).mean()
group_order = group_means.sort_values(['comb'], ascending = False).index
bin_ = np.arange(fcon.comb.min(), fcon.comb.max()+3, 3)
g = sns.FacetGrid(data=fcon, col = 'VClass', col_wrap=3,  col_order = group_order)
g.map(plt.hist, 'comb', bins=bin_)
g.set_titles('{col_name}')
plt.show()



ax = sns.barplot(x="VClass", y="comb", data=fcon,
        color=sns.color_palette()[0], ci="sd") # errwidth = 0
plt.xticks(rotation=15)
plt.ylabel("Average combined fuel efficiency (mgp)")
plt.show()

ax = sns.pointplot(x="VClass", y="comb", data=fcon) # linestyple = " "
plt.xticks(rotation=15)
plt.ylabel("Average combined fuel efficiency (mgp)")
plt.show()







# set bin edges, compute centers
bin_size = 0.25
xbin_edges = np.arange(0.5, fcon['displ'].max()+bin_size, bin_size)
xbin_centers = (xbin_edges + bin_size/2)[:-1]

# compute statistics in each bin
data_xbins = pd.cut(fcon['displ'], xbin_edges, right = False, include_lowest = True)
y_means = fcon['comb'].groupby(data_xbins).mean()
# y_sems = fcon['comb'].groupby(data_xbins).sem()

# plot the summarized data
plt.errorbar(x = xbin_centers, y = y_means)
plt.xlabel('displacement (l)')
plt.ylabel('CO2 (g\l)')
plt.show()