import argparse
import logging

import gym
import tensorflow as tf

from models.DDDQN import DDDQN
from trainers.DDDQNTrainer import DDDQNTrainer
from utilities import get_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config = get_config("config.json")

"""
# Using tf flags instead of argparse
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_episodes", args.num_episodes,
                            "Maximum number of episodes to train.")
"""
parser = argparse.ArgumentParser()

# Environment configurations
parser.add_argument(
    "--env", type=str, default=config.env,
    help="Environment name.")

# Netowrk configurations
parser.add_argument(
    "--learning-rate", type=float,
    default=config.learning_rate,
    help="Learning rate for the model.")

parser.add_argument(
    "--steps_per_target_update", type=int,
    default=config.steps_per_target_update,
    help="Number of steps after which to update target network weights.")

parser.add_argument(
    "--gamma", type=float, default=config.gamma,
    help="Discount factor.")

parser.add_argument(
    "--common_net_hidden_dimensions", type=int, nargs="+",
    default=config.common_net_hidden_dimensions,
    help="List of hidden layers dimensions for common network.")

parser.add_argument(
    "--buffer_size", type=float, default=config.buffer_size,
    help="Size of experience replay buffer.")

# Trainer configurations
parser.add_argument(
    "--num-episodes", type=int,
    default=config.num_episodes,
    help="Maximum number of episodes to train.")

parser.add_argument(
    "--max_checkpoints_to_keep", type=int,
    default=config.max_checkpoints_to_keep,
    help="Maximum number of recent checkpoints to store.")

parser.add_argument(
    "--mini_batch_size", type=int, default=config.mini_batch_size,
    help="Batch size.")

parser.add_argument(
    "--consecutive_successful_episodes_to_stop", type=int,
    default=config.consecutive_successful_episodes_to_stop,
    help="Consecutive number successful episodes above min avg to stop "
         "training.")

# Filesystem configurations
parser.add_argument(
    "--summray_dir", type=str, default=config.summray_dir,
    help="Directory path to store tensorboard summaries.")

parser.add_argument(
    "--save_after_num_episodes", type=int,
    default=config.save_after_num_episodes,
    help="Number of episodes after which to save our model.")

parser.add_argument(
    "--checkpoints_dir", type=str, default=config.checkpoints_dir,
    help="Directory path to store model checkpoints.")

parser.add_argument(
    "--load_existing", action="store_true",
    help="Load models from existing checkpoints in `checkpoints_dir`")

# Update the configuration
config = parser.parse_args()

sess = tf.Session()
env = gym.make(config.env)

config.input_size = len(env.reset())
config.output_size = env.action_space.n

main_network = DDDQN(scope_name="q_main", config=config)
target_network = DDDQN(scope_name="q_target", config=config)

if config.load_existing:
    main_network.load(sess)

trainer = DDDQNTrainer(sess, main_network, target_network, env, config, logger)

trainer.train()
