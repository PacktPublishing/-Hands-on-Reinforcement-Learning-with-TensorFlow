import tensorflow as tf


class BaseTrainer:
    def __init__(
            self, sess, main_network, target_network,
            env, config, logger):
        self.main_network = main_network
        self.target_network = target_network
        self.env = env
        self.logger = logger
        self.config = config
        self.sess = sess

        # Initialize variables to be logged after every episode
        self.current_loss = 0.0
        self.current_episode_reward = 0

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.writer = tf.summary.FileWriter(self.config.summray_dir)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        summary_writer_op = self.initialize_summary_writer_op()

        for ep_num in range(self.config.num_episodes):
            self.train_episode(ep_num)

            if ep_num % self.config.save_after_num_episodes == 0:
                self.main_network.save(self.sess, ep_num)

            self.logger.info(
                "Episode: {}  reward: {}  loss: {}  last_{}_avg_reward: {}"
                .format(ep_num, self.current_episode_reward,
                        self.current_loss,
                        self.config.consecutive_successful_episodes_to_stop,
                        self.last_n_avg_reward))

            if summary_writer_op is not None:
                self.append_summary(summary_writer_op, ep_num)

    def initialize_summary_writer_op(self):
        pass

    def append_summary(self, summary_writer_op, ep_num):
        pass

    def train_episode(self, ep_num):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
