import numpy as np
import pandas as pd
import operator
import pandas.io.data
import re

from sklearn import preprocessing
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.qda import QDA
from datetime import datetime
from dateutil import parser
from backtest import Strategy, Portfolio


def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  pd.io.data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    
    return df

def getStockFromQuandl(symbol, name, start, end):
    """
    Downloads Stock from Quandl.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    import Quandl
    df =  Quandl.get(symbol, trim_start = start, trim_end = end, authtoken="your token")
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + name
    df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
    
    return df

def getStockDataFromWeb(fout, start_string, end_string):
    """
    Collects predictors data from Yahoo Finance and Quandl.
    Returns a list of dataframes.
    """
    start = parser.parse(start_string)
    end = parser.parse(end_string)
    
    nasdaq = getStock('^IXIC', start, end)
    frankfurt = getStock('^GDAXI', start, end)
    london = getStock('^FTSE', start, end)
    paris = getStock('^FCHI', start, end)
    
    djia = getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start_string, end_string) 
    
    out =  pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq, djia, frankfurt, london, paris]



