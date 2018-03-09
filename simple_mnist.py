import tensorflow as tf
from tensorflow.contrib.timeseries.examples import predict
from tensorflow.examples.tutorials.mnist import input_data
import ssl

def add_layer(inputs, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    temp = tf.matmul(inputs, weight)
    wx_plus_b = temp + biases
    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)

# 全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)


with tf.name_scope('input'):
    w = tf.Variable(tf.zeros([784,10]) + 0.1, name='x_input')
    b = tf.Variable(tf.zeros([10]) + 0.1, name='y_input')

prediction = add_layer(x, 784, 10, tf.nn.softmax)

# l1 = add_layer(x, 784, 300, tf.nn.tanh)
# l1_drop = tf.nn.dropout(l1, keep_prob=keep_prob)

# prediction = add_layer(l1_drop, 300, 10, tf.nn.softmax)

# prediction = tf.nn.softmax(tf.matmul(x, w)+b)
# drop_out = tf.nn.dropout(prediction, keep_prob)

loss = tf.reduce_mean(tf.square(y-prediction))

# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for i in range(20):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_x,y:batch_y,keep_prob:1})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1})
        print('Iter' + str(i) + 'test accuracy:' + str(acc))
