import tensorflow as tf

data1 = [[1, 2], [3, 4]] #2x2
data2 = [[3, 6, 2], [4, 2, 4]] #2x3

m1 = tf.placeholder(tf.float32, [2, 2])
m2 = tf.placeholder(tf.float32, [2, 3])

product = tf.matmul(m1, m2) #2x3

with tf.Session() as sess:
	print(sess.run(product, feed_dict={m1: data1, m2: data2}))
