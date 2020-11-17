import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
import matplotlib.pyplot as plt

from printdescribe import print2, changepath

pth = r"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\GroupWork"

with changepath(pth):
    data_r = pd.read_excel("greece_quarterly_30Y_reduced_20201102.xlsx", sheet_name="Reduced")

print2(data_r.head())

data_r2 = data_r.iloc[2:,:].set_index("Name")
print2(data_r2.head(), data_r2.info())

data = data_r2.iloc[:, [0,10,12,27,9]]
print2(data.head)

# M4 Money Plot
money = wb.download(indicator='GC.TAX.TOTL.CN', country=['GR','GRC'], 
                    start=pd.to_datetime('1990', yearfirst=True), end=pd.to_datetime('2020', yearfirst=True)
                   , freq='Q')
money = money.reset_index()
print(money.head(10))
# money_plot = money.iloc[::-1,:].hvplot.line(x='year', y='FM.LBL.BMNY.GD.ZS', by='country', title='Broad money (% of GDP)')

data4 = data.reset_index()
data4[["Quarter", "Year"]] = data4['Name'].str.split(expand=True)
print2(data4.head())

result = pd.merge(data4, money, left_on="Year", right_on="year", how="outer")
cleandata = result.dropna()