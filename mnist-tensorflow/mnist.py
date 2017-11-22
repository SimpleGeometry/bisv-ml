import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=(784, 1))
y = tf.placeholder(tf.float32, shape=(10, 1))

W1 = tf.Variable(tf.random_normal(shape=(10, 784), stddev=2/794))
b1 = tf.Variable(tf.random_normal(shape=(10, 1)))

output = tf.nn.softmax(tf.add(tf.matmul(W1, X), b1))

cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(y, tf.negative(tf.log(output)))))

optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
  
  #train the code
	for i in range(10000):
		input_data, output_data = mnist.next_batch()
		_, loss = sess.run([optimizer, cost], feed_dict={X: input_data, y: output_data})
    
    
	percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))
	print(sess.run(percent_correct, feed_dict={X: mnist.test.images, y: mnist.test.labels}))
