import tensorflow as tf

x = tf.Variable(5, name="var_x")
y = tf.Variable(4, name="var_y")

product = tf.multiply(x, y)

all_saver = tf.train.Saver()
y_saver = tf.train.Saver({"var_y": y})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(product)
    all_saver.save(sess, './checkpoints/save_all')
    y_saver.save(sess, './checkpoints/save_y')
