import csv
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) #skip column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_prices_SVR(dates, prices, x):
		# convert to np array of n X 1
		dates = np.reshape(dates,(len(dates), 1))
		prices = np.reshape(prices, (len(prices), 1))

		svr_lin = SVR(kernel = 'linear', C = 1e3)
		svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
		svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

		svr_lin.fit(dates, prices)
		svr_poly.fit(dates, prices)
		svr_rbf.fit(dates, prices)
		'''
		plt.scatter(dates, prices, color = 'black', label = 'Data')
		plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'RBF Model')
		plt.plot(dates, svr_lin.predict(dates), color = 'green', label = 'Linear Model')
		plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'Poly Model')
		plt.xlabel('Dates')
		plt.ylabel('Prices')
		plt.title('Support Vector Regression')
		plt.show()
		'''
		return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_prices_LR(dates,prices,x):
		# convert to np array of n X 1
		dates = np.reshape(dates,(len(dates),1))
		prices = np.reshape(prices,(len(prices),1))
		
		linear_mod = linear_model.LinearRegression() 
		linear_mod.fit(dates,prices) #fit data points to the model
		predicted_price = linear_mod.predict(x)
		'''
		plt.scatter(dates,prices, color='purple') 
		plt.plot(dates,linear_mod.predict(dates),color='red',linewidth=3) #plot LR line 
		plt.show()
		'''
		return predicted_price[0][0],linear_mod.coef_[0][0] ,linear_mod.intercept_[0]

get_data('tsla.csv') 

predicted_price_LR, coefficient, constant = predict_prices_LR(dates, prices, 29)  
print "The stock open price upcoming: $", str(predicted_price_LR)
print "The regression coefficient is ", str(coefficient),", and the constant is ", str(constant)

predicted_price_SVR = predicted_price_SVR(dates, prices, 29)
print "The stock open price found with SVR: $" str(predicted_price_SVR)
