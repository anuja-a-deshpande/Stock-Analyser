import csv
import numpy as np
from sklearn.svm import SVR

stock1_train = []
stock2_train = []

def get_data(filename, stock1, stock2):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			stock1.append(float(row[0]))
			stock2.append(float(row[1]))
	return

def predict_price(stock1, stock2, x):
	stock1 = np.reshape(stock1,(len(stock1), 1)) # converting to matrix of n X 1
	x = np.reshape(x,(len(x), 1)) # converting to matrix of n X 1
	# svr_lin = SVR(kernel= 'linear', C= 1e3)
	# svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3) # defining the support vector regression models
	svr_rbf.fit(stock1, stock2) # fitting the data points in the models
	# svr_lin.fit(stock1, stock2)
	# svr_poly.fit(stock1, stock2)
	return svr_rbf.predict(x)

get_data('stock data - Train3.csv', stock1_train, stock2_train) # calling get_data method by passing the csv file to it

stock1_test = []
stock2_test = []
get_data('stock data - Test3.csv', stock1_test, stock2_test)

print(stock1_test)
print(stock2_test)

predicted_price = predict_price(stock1_train, stock2_train, stock1_test) 

print(predicted_price)