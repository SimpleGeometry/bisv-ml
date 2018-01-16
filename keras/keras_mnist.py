#imports
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils


#data processing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(60000, 784)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#model declaration and compilation
model = Sequential()
model.add(Dense(units=10, activation='softmax', input_dim=784))

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


#model training and evaluation
model.fit(X_train, y_train, epochs=5, batch_size=32)
loss, acc = model.evaluate(X_test, y_test)

print('Final accuracy:', acc)
