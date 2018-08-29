import os

import tensorflow as tf


class BaseModel:
    def __init__(self, scope_name, config):
        self.scope_name = scope_name
        self.config = config
        self.saver = None

    def save(self, sess, global_step):
        self.saver.save(
            sess,
            self.config.checkpoints_dir + os.sep + "/model.ckpt",
            global_step)

    def load(self, sess):
        latest_checkpoint = (
            tf.train.latest_checkpoint(self.config.checkpoints_dir))
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)

    def initialize_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
