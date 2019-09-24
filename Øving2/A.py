import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

epochs = 100

x_values = [[1], [0]]
y_values = [[0, 1], [1, 0]]

X = tf.placeholder(dtype = tf.float32 ,shape = [None, 1], name = "X_placeholder")
y_true = tf.placeholder(dtype = tf.float32 ,shape = [None, 2], name = "X_placeholder") 

values = np.random.uniform(low=-0.5, high=0.5, size=(1,1))

W = tf.Variable(values, dtype = tf.float32, shape = [1, 1])
b = tf.Variable(values, dtype = tf.float32, shape = [1, 1])

logits = tf.add(tf.multiply(X,W),b)
loss = tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y_values)
init = tf.global_variables_initializer();

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000001)

train = optimizer.minimize(loss)


with tf.Session() as sess:
	sess.run(init)

	for i in range(x_values):
		rand_ind = np.random.randint(low = 0, high = len(x_values))

		feed = {X: x_values[rand_ind], y_true: y_values}
		sess.run(train, feed_dict = feed)

	loss_value = loss.eval(feed_dict=feed)
    model_w = sess.run(W)
    model_b = sess.run(b)

print(model_W)
print(model_b)








