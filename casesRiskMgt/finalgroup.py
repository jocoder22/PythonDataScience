import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
import matplotlib.pyplot as plt
from statsmodels.formula.api import rlm
import statsmodels.api as sm

from printdescribe import print2, changepath
pth = r"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\GroupWork"

# Access Greece economic data
with changepath(pth):
    data_r = pd.read_excel("greece_quarterly_30Y_reduced_20201102.xlsx", sheet_name="Reduced")

print2(data_r.head())
data_r2 = data_r.iloc[2:,:].set_index("Name")
print2(data_r2.head(), data_r2.info())

data = data_r2.iloc[:, [0,10,12,27,9]]
print2(data.head)

# Download Greece Tax revenue data
money = wb.download(indicator='GC.TAX.TOTL.CN', country=['GR','GRC'], 
                    start=pd.to_datetime('1990', yearfirst=True), end=pd.to_datetime('2020', yearfirst=True)
                   , freq='Q')
money = money.reset_index()
print(money.head(10))
# money_plot = money.iloc[::-1,:].hvplot.line(x='year', y='FM.LBL.BMNY.GD.ZS', by='country', title='Broad money (% of GDP)')

# split the Name (quarter  year) and form columns with each split
data4 = data.reset_index()
data4[["Quarter", "Year"]] = data4['Name'].str.split(expand=True)
print2(data4.head())

# Merge the two dataset
result = pd.merge(data4, money, left_on="Year", right_on="year", how="outer")
cleandata = result.dropna()

# Select columns for plotting
graphing = cleandata.iloc[:,[0,1,2,3,4,5,10]]
colnames = ["Time", "GDP", "CPI","InterBank Rate", "M3 Outstanding", "Govt Bond-15yr", "Tax Revenue"]
graphing.columns = colnames
graphing = graphing.set_index("Time")

# plot the datasets
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 6))
color = ["r", "g"]
p = ax.flatten().tolist()
graphing2 = graphing.iloc[:, [0,2]]
for indx, colname in enumerate(graphing2.columns):
    ba =  indx
    graphing[colname].plot(title = f'{colname}', ax=p[ba], color=color[indx])
    fig.subplots_adjust(hspace=.3)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 6))
p = ax.flatten().tolist()
graphing3 = graphing.iloc[:, [1,3]]
for indx, colname in enumerate(graphing3.columns):
    ba =  indx
    graphing[colname].plot(title = f'{colname}', ax=p[ba], color=color[indx])
    fig.subplots_adjust(hspace=.3)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 6))
p = ax.flatten().tolist()
graphing4 = graphing.iloc[:, [4,5]]
for indx, colname in enumerate(graphing4.columns):
    ba =  indx
    graphing[colname].plot(title = f'{colname}', ax=p[ba], color=color[indx])
    fig.subplots_adjust(hspace=.3)

plt.show()

# y => gdp , X => the rest of dataset
y = cleandata.iloc[:,[1]]
X = cleandata.iloc[:,[2,3,4,5,10]]

# Add constant
Xk = sm.add_constant(X)

# Build linear regression model
est = sm.OLS(y.astype(float), Xk.astype(float))

# Estimate coefficients and print the summary
est2 = est.fit()
print2(est2.summary())


# Build robust linear regression model
rlm_model = sm.RLM(y.astype(float), Xk.astype(float),  M=sm.robust.norms.HuberT())

# Estimate coefficients and print the summary
est2 = rlm_model.fit()
print(est2.summary())

XX2 = cleandata.iloc[:,[3,4, 5,10]]
Xk2 = sm.add_constant(XX2)

rlm_model2 = sm.RLM(y.astype(float), Xk2.astype(float),  M=sm.robust.norms.HuberT())
est22 = rlm_model2.fit()
print(est22.summary())
