import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel


class DDDQN(BaseModel):
    def __init__(self, scope_name, config):
        super(DDDQN, self).__init__(scope_name, config)

        self.build_model()
        self.initialize_saver()

    def build_model(self):
        with tf.variable_scope(self.scope_name):
            self.input = tf.placeholder(shape=[None, self.config.input_size],
                                        dtype=tf.float32)

            # Logic for common hidden network
            common_net = self.input
            for layer_size in self.config.common_net_hidden_dimensions:
                common_net = tf.layers.dense(common_net,
                                             layer_size,
                                             activation=tf.nn.relu)

            # Separating streams into advantage and value networks
            adv_net = tf.layers.dense(common_net, 32, activation=tf.nn.relu)
            adv_net = tf.layers.dense(adv_net, self.config.output_size)

            val_net = tf.layers.dense(common_net, 32, activation=tf.nn.relu)
            val_net = tf.layers.dense(val_net, 1)

            self.output = val_net + (adv_net - tf.reduce_mean(adv_net,
                                                              axis=1,
                                                              keepdims=True))

            # Placeholder for expected q-values
            self.y = tf.placeholder(shape=[None, self.config.output_size],
                                    dtype=tf.float32)

            # Using the loss method provided by tf directly
            self.loss = tf.losses.mean_squared_error(self.y, self.output)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

    def initialize_saver(self):
        self.saver = tf.train.Saver(
            tf.global_variables(),
            max_to_keep=self.config.max_checkpoints_to_keep)

    def predict(self, session, state):
        return session.run(
            self.output,
            feed_dict={
                self.input: np.reshape(state, [-1, self.config.input_size])
            })

    def update(self, session, state, y):
        return session.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.input: state,
                self.y: y
            })

    @staticmethod
    def create_copy_operations(source_scope, dest_scope):
        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=source_scope)
        dest_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=dest_scope)

        assert len(source_vars) == len(dest_vars)

        result = []

        for source_var, dest_var in zip(source_vars, dest_vars):
            result.append(dest_var.assign(source_var.value()))

        return result
