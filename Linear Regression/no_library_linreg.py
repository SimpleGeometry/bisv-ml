#import necessary libraries
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

#calculates result
def h(b, w, x):
	result = b
	for weight, feature in zip(w, x):
		result += weight * feature
	return result

#gradient descent step
def update(b, w, x, y, alpha):
	w_new = w[:]
	b_new = b
	for i in range(len(w)):
		w_new[i] -= alpha * (h(b, w, x) - y) * x[i]
	b_new -= alpha * (h(b, w, x))
	return w_new, b_new

#returns model cost
def J(b, w, x, y):
	return ((h(b, w, x) - y) ** 2) / 2

#getting datasets into nice form
boston = datasets.load_boston()
X = boston.data #shape of the dataset is 506, 13
y = boston.target #shape is 506, 1

#split data into training and test
X_train = X[:-100]
X_test = X[-100:]
y_train = y[:-100]
y_test = y[-100:]

#model parameters
b = 0
w = []
for i in range(13):
	w.append(0)

alpha = 0.00000003

#training the model
costs = []

steps = 10000
log_steps = 100
cost = 0
for i in range(steps):
	cur_X = X_train[i % X_train.shape[0]]
	cur_y = y_train[i % y_train.shape[0]]
	cost += J(b, w, cur_X, cur_y)
	w, b = update(b, w, cur_X, cur_y, alpha)
	#store the costs so we can plot a function later
	if i % log_steps == 0:
		costs.append(cost)
		cost = 0
	#shuffle data every time we go through all the data points once
	if i % X_train.shape[0] == 0:
		np.random.shuffle(X_train)
		np.random.shuffle(y_train)

#testing the model
predictions = []
for i in range(X_test.shape[0]):
	predictions.append(h(b, w, X_test[i]))

#plots prediction (the features are multidimensional so it's not a straight line)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(X_test, y_test, 'bo')
plt.plot(X_test, predictions, 'r-')
plt.xlabel('Feature')
plt.ylabel('Price ($10000s of dollars)')
plt.title('House Price Regression (Test Data)')
plt.grid(True)

#plots the cost function over training iterations
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, steps, log_steps), costs, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)

plt.tight_layout() #prevent text overlapping

plt.show()
