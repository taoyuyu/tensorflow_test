# coding:utf-8

import tensorflow as tf
import input_data
import numpy as np
import platform as plt

# x是特征值
x = tf.placeholder("float", [None, 784])
# w表示每一个特征值（像素点）会影响结果的权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y即模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

mnist = input_data.read_data_sets("./train/", one_hot=True)

# 循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

for i in range(0, len(mnist.test.images)):
    result = sess.run(correct_prediction,
                      feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
    if not result:
        print(
        'predict value：', sess.run(y, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
        print(
        'real value：', sess.run(y_, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
        one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
        pic_matrix = np.matrix(one_pic_arr, dtype="float")
        plt.imshow(pic_matrix)
        # pylab.show()
        break

print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

