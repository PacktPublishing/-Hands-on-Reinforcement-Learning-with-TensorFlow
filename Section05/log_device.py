import tensorflow as tf

matrix_1 = tf.constant([[3, 5]])
matrix_2 = tf.constant([[2], [4]])

product = tf.matmul(matrix_1, matrix_2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(product))
