import random
from collections import deque

import numpy as np
import tensorflow as tf

from trainers.BaseTrainer import BaseTrainer


class DDDQNTrainer(BaseTrainer):
    def __init__(self, sess, main_network, target_network,
                 env, config, logger):

        super(DDDQNTrainer, self).__init__(sess, main_network, target_network,
                                           env, config, logger)

        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.last_n_rewards = deque(
            maxlen=config.consecutive_successful_episodes_to_stop)

    @property
    def last_n_avg_reward(self):
        return np.mean(self.last_n_rewards)

    def initialize_summary_writer_op(self):
        self.reward_placeholder = tf.placeholder('float32')
        self.avg_placeholder = tf.placeholder('float32')
        self.loss_placeholder = tf.placeholder('float32')

        tf.summary.scalar('loss', self.loss_placeholder)
        tf.summary.scalar('episode_reward', self.reward_placeholder)
        tf.summary.scalar('last_n_avg_reward', self.avg_placeholder)

        return tf.summary.merge_all()

    def append_summary(self, summary_writer_op, ep_num):

        summary = self.sess.run(summary_writer_op, {
            self.loss_placeholder: self.current_loss,
            self.reward_placeholder: self.current_episode_reward,
            self.avg_placeholder: self.last_n_avg_reward
        })

        self.writer.add_summary(summary, ep_num)
        self.writer.flush()

    def train_episode(self, ep_num):
        state = self.env.reset()
        done = False
        self.current_episode_reward = 0
        steps = 0

        # epsilon decay
        epsilon = 1. / ((ep_num / 10) + 1)

        while not done:
            # select the action
            action = None
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.main_network.predict(self.sess, state))

            # execute the action
            next_state, reward, done, _ = self.env.step(action)

            if done:
                reward = -1

            # add to the buffer
            self.replay_buffer.append(
                (state, action, reward, next_state, done))

            if len(self.replay_buffer) < self.config.mini_batch_size:
                continue

            self.current_loss = self.train_step()

            if steps % self.config.steps_per_target_update == 0:
                self.sess.run(
                    self.main_network.create_copy_operations(
                        self.main_network.scope_name,
                        self.target_network.scope_name))

            self.current_episode_reward += reward
            steps += 1
            state = next_state

        self.last_n_rewards.append(self.current_episode_reward)

    def train_step(self):
        mini_batch = random.sample(self.replay_buffer,
                                   self.config.mini_batch_size)
        states = [x[0] for x in mini_batch]
        states = np.vstack(states)

        actions = np.array([x[1] for x in mini_batch])
        rewards = np.array([x[2] for x in mini_batch])
        next_states = np.vstack([x[3] for x in mini_batch])
        done = np.array([x[4] for x in mini_batch])

        target_output_next_states = self.target_network.predict(
            self.sess, next_states)

        # For double DQN: select the best action for next state
        main_output_next_states = self.main_network.predict(
            self.sess, next_states)

        selected_best_actions = np.argmax(main_output_next_states, axis=1)
        target_output_for_selected_actions = target_output_next_states[
            np.arange(len(states)), selected_best_actions]

        target_q_vals = (
            rewards + self.config.gamma *
            target_output_for_selected_actions *
            (1 - done))

        main_output = self.main_network.predict(self.sess, states)
        main_output[np.arange(len(states)), actions] = target_q_vals

        loss, optimizer = self.main_network.update(
            self.sess, states, main_output)

        return loss
