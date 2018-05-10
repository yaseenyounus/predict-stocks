import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		FileReader = csv.reader(csvfile)
		
		next(FileReader)
		
		for row in FileReader:
			dates.append(int(row[0].split('/')[1]))
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) 

	lin_model = SVR(kernel= 'linear', C= 1e3)
	poly_model = SVR(kernel= 'poly', C= 1e3, degree= 2)
	rbf_model = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 

	lin_model.fit(dates, prices)
	poly_model.fit(dates, prices)
	rbf_model.fit(dates, prices) 

	plt.scatter(dates, prices, color= 'black', label= 'Data') 

	plt.plot(dates, rbf_model.predict(dates), color= 'red', label= 'RBF model') 
	plt.plot(dates, lin_model.predict(dates), color= 'green', label= 'Linear model') 
	plt.plot(dates, poly_model.predict(dates), color= 'blue', label= 'Polynomial model') 

	plt.xlabel('Date (March 2018)')
	plt.ylabel('Price ($)')
	plt.title('Amazon Stock Prices')
	plt.legend()
	
	plt.show()

	return rbf_model.predict(x)[0], lin_model.predict(x)[0], poly_model.predict(x)[0]

get_data('AMZN.csv') 

prediction = predict_price(dates, prices, 30)


