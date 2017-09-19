from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

boston = datasets.load_boston()
X = boston.data[:, 5]
X = X.reshape((X.shape[0], 1))
y = boston.target

X_train = X[:-100]
X_test = X[-100:]
y_train = y[:-100]
y_test = y[-100:]

###########################################
# Functions to run the program are below. Don't scroll down if you're trying to figure out the correct calls yourself



















h = linear_model.SGDRegressor(penalty='none', eta0=0.03, n_iter=100000)
h.fit(X_train, y_train)

print('weights:', h.coef_)
print('bias:', h.intercept_)
print('r^2:', h.score(X, y))


#Graph plotting
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(X_test, y_test, 'bo')
plt.plot(X_test, h.predict(X_test), 'r-')
plt.xlabel('Feature')
plt.ylabel('Price ($10000s of dollars)')
plt.title('House Price Regression (Test Data)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(X_train, y_train, 'bo')
plt.plot(X_train, h.predict(X_train), 'r-')
plt.xlabel('Feature')
plt.ylabel('Price ($10000s of dollars)')
plt.title('House Price Regression (Training Data)')
plt.grid(True)

plt.tight_layout() #prevent text overlapping

plt.show()
