#!/usr/bin/env python
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime

# startdate = datetime(2015, 1, 1)
startdate = datetime(2018, 9, 12)
enddate = date.today()

folders = ['Apple', 'MicroSoft', 'Google', 'Netflix', 'Tesla', 'Amazon', 'Toyota', 'JPMorgan', 
            'Citigroup', 'Walmat', 'Target', "Fedex", "Ups", "Walgreens", "Disney", "Pfizer",
            "Cvs", "AT_T", "CocaCola", "Boeing", "SolarEdge", "AdvancedMicroDevices", "Twilio",
            "ExpWorld", "HomeDepot", "Ford", "PVH", "Twitter", "Salesforce", 
            "Alibaba", "NioElectricMotor", "Apple" ,  "BristolMyers"]

symbols = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'TSLA', 'AMZN', 'TM', 'JPM', 'C', 'WMT', 'TGT', 'FDX',
            'UPS', 'WBA', 'DIS', 'PFE', 'CVS', 'T', 'KO', 'BA', 'SEDG', 'AMD', 'TWLO', 'EXPI', 'HD',
              'F', 'PVH', 'TWTR', 'CRM', 'BABA', 'NIO', 'AAPL', 'BMY']

coldict = {symbols[i]:folders[i] for i in range(len(symbols))}

df = pdr.get_data_yahoo(symbols, startdate, enddate)['Adj Close']
df.head()
