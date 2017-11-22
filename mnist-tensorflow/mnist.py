import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#it's None instead of 1 for the first dimension so you can train the data as a "batch" 
#instead of one data point at a time (training is faster) [i explain later]
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=2/794))
b = tf.Variable(tf.random_normal(shape=[1, 10]))

output = tf.nn.softmax(tf.matmul(X, W) + b)

#compute the cross entropy (the difference) between the output(prediction) and the actual value
cost = -tf.reduce_sum(y * tf.log(output))

optimizer = tf.train.AdamOptimizer().minimize(cost)



training_steps = 1000
#to evaluate accuracy (optional)
percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
  
  	#train the model (update weights and biases)
	for i in range(training_steps):
		input_data, output_data = mnist.train.next_batch(batch_size=100)
		sess.run(optimizer, feed_dict={X: input_data, y: output_data})

		#print accuracy of the model every 20 steps
		if i % 20 == 0:
			print('Accuracy at step', i, ':', sess.run(percent_correct, feed_dict={X: mnist.test.images, y: mnist.test.labels}))
