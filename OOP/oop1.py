#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
# plt.style.use('ggplot')

class filelink:
    """  """
    def __init__(self, filename):
        self.filename = filename



class data_analysis(filelink):
    """   """
    def __init__(self, filename):
        super().__init__(filename)
        # filelink.filename = filename
        # filelink.__init__(self, filename) # for multiple inheritance
        self.data = pd.read_csv(self.filename)
        self.stats = self.data.describe()



    def get_eda(self):
        shape = self.data.shape
        head = self.data.head()
        colnames =  self.data.columns
        return shape, head, colnames

sp = {'sep':'\n\n', 'end':'\n\n'}
url = 'https://assets.datacamp.com/production/repositories/2097/datasets/5dd3a8250688a4f08306206fa1d40f63b66bc8a9/us_life_expectancy.csv'

# data = pd.read_csv(url)

data1 = data_analysis(url)
# data1.csvdata()
s, h, c = data1.get_eda()
print(s, h, c, **sp)
print(f'Method Resolution Order (MRO) is:\n {data_analysis.__mro__}', **sp)

print(data1.stats, **sp)


