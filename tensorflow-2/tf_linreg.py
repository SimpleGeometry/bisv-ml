#import libraries
import tensorflow as tf
import numpy as np

def get_data():
	#data is from the computer hardware dataset found on the UCI ML repository
	with open('data.txt', 'r') as fin:
		text_in = fin.read()

	split = text_in.splitlines()
	data = []
	for line in split:
		data.append(line.split(','))
	np_data = np.array(data)
	x = np_data[:, 2:8].astype('f4')
	y = np_data[:, 8].astype('f4')

	#normalize features of x
	x_mean = np.mean(x, 0)
	x_std = np.std(x, 0)
	x = (x - x_mean) / x_std

	return x, y

def tf_summary():
	if tf.gfile.Exists("summary"):
		tf.gfile.DeleteRecursively("summary")

	tf.summary.scalar('cost', cost)
	tf.summary.histogram('weights', w)
	tf.summary.histogram('bias', b)

	summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter("summary")
	writer.add_graph(sess.graph)
	return summary, writer

#get data
x_data, y_data = get_data()
n_examples = np.shape(x_data)[0]
n_features = np.shape(x_data)[1]

x_data = np.transpose(x_data)
y_data = np.reshape(y_data, [1, n_examples])


##############################  YOUR CODE HERE  #####################################

''' Replace all the quotes/variables in quotes with the correct code '''

#declare graph
#1: declare placeholders x and y (to hold data)
x = 'x'
y = 'y'

#2: declare variables w (weights) and b (bias)
w = 'w'
b = 'b'

#3: declare operations and output (multiplication)
h = 'h'

#declare cost function
cost = 'cost'

#declare optimizer and learning rate
learning_rate = 'learning rate'
optimizer = 'optimizer'

#run graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	#tensorboard stuff
	summary, writer = tf_summary()

	#train model
	iterations = 'iterations'
	for i in range(iterations):
		#fill in var1, 2, 3 with the correct code
		sess.run('var1', feed_dict={x: 'var2', y: 'var3'})

		#this is for logging the results to tensorboard so you can visualize them (i % 10 == 0 says to log the result every 10 iterations)
		if i % 10 == 0:
			writer.add_summary(sess.run(summary, feed_dict={x: 'var2', y: 'var3'}))
