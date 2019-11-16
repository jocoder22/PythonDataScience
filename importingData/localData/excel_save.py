import os
import pandas as pd
import csv
from openpyxl import Workbook

path = r"D:\PythonDataScience\shellDataProcessing"
os.chdir(path)
print(os.getcwd())


df_new = pd.read_csv('Spotify201809.csv')
writer = pd.ExcelWriter('newfile.xlsx')
df_new.to_excel(writer, index = False)
writer.save()


