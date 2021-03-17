import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime

startdate = datetime(2015, 1, 1)
enddate = date.today()

tickers = 'FB AMZN AAPL GOOGL NFLX MSFT ^GSPC'.split()
colname = 'Apple Amazon Facebook  Google Microsoft Netflix S&P500'.split()


###################
ticker2 = sorted('FB AMZN AAPL GOOGL NFLX MSFT ^GSPC'.split())
colname2 = sorted('Facebook  Google Apple Amazon Microsoft Netflix S&P500'.split())

newdict = {ticker[i]:colname[i] for i in range(len(colname2))}

portfolio = pdr.get_data_yahoo(tickers, startdate, enddate)['Adj Close']
portfolio.columns = colname
print(portfolio.head())

portfolio.iloc[:,[0,2,4]].plot(figsize=(14,8))
plt.grid(color='black', which='major', axis='y', linestyle='solid')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3);

portfolio.iloc[:,[1,3,5,6]].plot(figsize=(14,8))
plt.grid(color='black', which='major', axis='y', linestyle='solid')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=4);


def stock_analysis(df):
    startdate = "2012-01-01"
    labels = ["Adjusted Close", "Volume Traded", "Value Traded"]
    
    if type(df) == str:
        stock = pdr.get_data_yahoo(df, startdate)
        stock['valueTraded'] = stock.Volume *  stock['Adj Close']
        
        fig, axs = plt.subplots(3, sharex=True, figsize=(14,12))
        plt.grid(color='black', which='major', axis='y', linestyle='solid')

        stock.loc[:,['Adj Close']].plot(ax=axs[0], label="Adjusted Close", title=f'{labels[0]} for {df}', grid=True)
        stock.loc[:,['Volume']].plot(ax=axs[1], label="Volume Traded", title=f'{labels[0]} for {df}', grid=True)
        stock.loc[:,['valueTraded']].plot(ax=axs[2], label="Value Traded", title=f'{labels[0]} for {df}', grid=True)
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
#         plt.grid(color='black', which='major', axis='y', linestyle='solid')
        plt.show()
        print("\n\n\n")
          
        return stock.head()

        
    elif isinstance(df, list):
        portfolio = pdr.get_data_yahoo(df, startdate)
        
        for name in df:
            fig, axs = plt.subplots(3, sharex=True, figsize=(14,12))
            
            portfolio['Adj Close'][name].plot(ax=axs[0], label=labels[0], title=f'{labels[0]} for {name}', grid=True)
            portfolio['Volume'][name].plot(ax=axs[1], label=labels[1], title=f'{labels[1]} for {name}', grid=True)
            valueTraded = portfolio['Volume'][name] * portfolio['Adj Close'][name]
            valueTraded.plot(ax=axs[2], label=labels[2], title=f'{labels[2]} for {name}', grid=True)
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
            print("\n\n\n")
         
          return portfolio.head()
                    
stock_analysis('AAPL')  

stock_analysis(tickers)
