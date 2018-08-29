import time

import numpy as np
import tensorflow as tf

x = tf.placeholder("float32")
y = tf.placeholder("float32")

tf.summary.scalar("tag_x", x)
tf.summary.histogram("tag_y", y)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./summary")
    writer.add_graph(sess.graph)

    for i in range(1000):
        time.sleep(0.1)
        summary = sess.run(merged_summary_op, {
            x: np.random.rand(),
            y: np.random.rand()
        })

        writer.add_summary(summary, i)
        writer.flush()
