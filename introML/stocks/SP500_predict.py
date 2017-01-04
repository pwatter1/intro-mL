from __future__ import division 
from datetime import datetime
from yahoo_finance import Share

import matplotlib.pyplot as plt
import graphlab as gl

# download historical prices of SP500
today = datetime.strftime(datetime.today(), '%Y-%m-%d')
stock = Share('^GSPC') # yahoo finance symbol for SP500 index
hist_quotes = stock.get_historical('2001-01-01', today)

l_date = []
l_open = []
l_high = []
l_low = []
l_close = []
l_volume = []

hist_quotes.reverse()

for quotes in hist_quotes:
        l_date.append(quotes['Date'])
        l_open.append(float(quotes['Open']))
        l_high.append(float(quotes['High']))
        l_low.append(float(quotes['Low']))
        l_close.append(float(quotes['Close']))
        l_volume.append(int(quotes['Volume']))

qq = gl.SFrame({'datetime' : l_date,
                'open'     : l_open,
                'high'     : l_high,
                'low'      : l_low,
                'close'    : l_close,
                'volume'   : l_volume})

# datetime is a string, so convert into datetime object
qq['datetime'] = qq['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

qq.save("SP500_daily.bin")
# retrieve
qq = gl.SFrame("SP500_daily.bin/")

# adding our outcome variable, 1 if upday, -1 if down day
qq['outcome'] = qq.apply(lambda x: 1 if x['close'] > x['open'] else -1)
# also adding 3 new columns for backtesting, 'ho', 'lo', 'gain'
qq['ho'] = qq['high'] - qq['open']
qq['lo'] = qq['low'] - qq['open']
qq['gain'] = qq['close'] - qq['open']

# lag our data
ts = gl.TimeSeries(qq, index='datetime')
ts['outcome'] = ts.apply(lambda x: 1 if x['close'] > x['open'] else -1)

ts_1 = ts.shift(1) #lag by x days
ts_2 = ts.shift(2)
ts_3 = ts.shift(3)

ts['feature1'] = ts['close'] > ts_1['close']
ts['feature2'] = ts['close'] > ts_2['close']
ts['feature3'] = ts['close'] > ts_3['close']

l_features = [ts['feature1'], ts['feature2'], ts['feature3']]

# ADD MORE FEATURES TO INCREASE ACCURACY OF MODEL

ts['gain'] = ts['close'] - ts['open']

ratio = 0.8
training = ts.to_sframe()[0:round(len(ts) * ratio)]
testing = ts.to_sframe()[round(len(ts) * ratio):]

max_tree_depth = 6
decision_tree = gl.decision_tree_classifier.create(training, validation_set=None,
													target='outcome', features=l_features,
													max_depth=max_tree_depth, verbose=False)
print decision_tree.evaluate(training)['accuracy'], decision_tree.evaluate(testing)['accuracy']

predictions = decision_tree.predict(testing)
# add the prediction column to testing set
testing['predictions'] = predictions
# check first ten predictions compared to real values
print testing[['datetime', 'outcome', 'predictions']].head(10)
