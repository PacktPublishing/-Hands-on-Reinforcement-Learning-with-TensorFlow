import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

learning_rate = 0.01
num_epochs = 1000

train_X = np.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
     7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])

train_Y = np.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
     2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples = train_X.shape[0]

input_x = tf.placeholder("float")
actual_y = tf.placeholder("float")

# Simple linear regression tries to find W and b such that
# y = Wx + b
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

prediction = tf.add(tf.multiply(input_x, W), b)

loss = tf.squared_difference(actual_y, prediction)
# loss = tf.Print(loss, [loss], 'Loss: ', summarize=n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess = tf_debug.TensorBoardDebugWrapperSession(
        # sess, 'localhost:6064')
    sess.run(init)

    initial_loss = sess.run(loss, feed_dict={
        input_x: train_X,
        actual_y: train_Y
    })
    print("Initial loss", initial_loss)

    for epoch in range(num_epochs):

        for x, y in zip(train_X, train_Y):
            _, c_loss = sess.run([optimizer, loss], feed_dict={
                input_x: x,
                actual_y: y
            })

    tf.add_to_collection("Asserts", tf.assert_less(loss, 2.0, [loss]))
    tf.add_to_collection("Asserts", tf.assert_positive(loss, [loss]))
    assert_op = tf.group(*tf.get_collection('Asserts'))

    final_loss, _ = sess.run([loss, assert_op], feed_dict={
        input_x: train_X,
        actual_y: train_Y
    })

    print("Final Loss: {}\n W:{}, b:{}".format(
        final_loss, sess.run(W), sess.run(b)))
