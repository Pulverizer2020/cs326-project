import numpy as np
import pandas as pd

df_sales = pd.read_csv('data/sales.csv')
df_products = pd.read_csv('data/products.csv')
df_orders = pd.read_csv('data/orders.csv')
df_customers = pd.read_csv('data/customers.csv')

def squareMatrix(matrix):
	squareMatrix = np.matrix.copy(matrix)
	rows = len(matrix)
	cols = len(matrix[0])

	for customerID in rows:
		for productID in cols:
			if squareMatrix[customerID][productID] != np.nan:
				squareMatrix[customerID][productID] = np.square(squareMatrix[customerID][productID])
	return squareMatrix

def logMatrix(matrix):
	logMatrix = np.matrix.copy(matrix)
	rows = len(matrix)
	cols = len(matrix[0])

	for customerID in rows:
		for productID in cols:
			price = df_products[df_products['product_ID'] == productID]['price']
			if squareMatrix[customerID][productID] != np.nan:
				logMatrix[customerID][productID] = logMatrix[customerID][productID] * np.log(price)
	return logMatrix