import tensorflow as tf

saver = tf.train.import_meta_graph("checkpoints/save_all.meta")
graph = tf.get_default_graph()

with tf.Session() as sess:
    saver.restore(sess, "checkpoints/save_all")
    var_y = graph.get_tensor_by_name("var_y:0")
    print(sess.run(var_y))
