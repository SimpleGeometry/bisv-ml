import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=1e-3))
b = tf.Variable(tf.random_normal(shape=[1, 10]))
output = tf.nn.softmax(tf.matmul(X, W) + b)

#compute the difference between output and y
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
learning_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())


	training_steps = 1000000
	for i in range(training_steps):
		input_data, output_data = mnist.train.next_batch(batch_size=100)
		sess.run(optimizer, feed_dict={X: input_data, y: output_data})

		#print accuracy of the model every 20 steps
		if i % 20 == 0:
			print('Accuracy at step', i, ':', sess.run(percent_correct, feed_dict={X: mnist.test.images, y: mnist.test.labels}))

