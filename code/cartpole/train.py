import argparse
import numpy as np
import gym
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'], default='CartPole-v0')
parser.add_argument('--evaluate_freq', type=int, default=10, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = env.unwrapped
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config)
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    rewards = []
    step = 0
    start = time.time()
    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        reward_eps = 0
        while not done:
            # env.render()
            time.sleep(0.05)
            # TODO: Get action from DQN.
            action = dqn.act(obs)

            # Act in the true environment.
            obs_, reward, done, info = env.step(action)

            # Preprocess incoming observation.
            x, x_dot, theta, theta_dot = obs_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = float(r1 + r2)

            #
            # if done:
            #   reward = -1

            if done:
              isdone = 0
            else:
              isdone = 1

            obs_ = preprocess(obs_, env=args.env).unsqueeze(0)

            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            memory.push(obs, torch.tensor(action), obs_, torch.tensor(reward),torch.tensor(isdone))

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
            step = step + 1
            obs = obs_
            reward_eps += 1
        rewards.append(reward_eps)
        # Evaluate the current agent.
        # if episode % args.evaluate_freq == 0:
        #     mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
        #
        #     print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')
        #     rewards.append(mean_return)
        #     # Save current agent if it has the best performance so far.
        #     if mean_return >= best_mean_return:
        #         best_mean_return = mean_return
        #
        #         print('Best performance so far! Saving model.')
        #         torch.save(dqn, f'models/{args.env}_best.pt')
        print("eps:%s" % episode, reward_eps)

    mean = np.mean(rewards)

    # for i in range(rewards):
    #   if rewards[i] > 3*mean:
    #     i = mean
    #

    plt.plot(np.arange(0, env_config['n_episodes'],1), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("exp5.png")
    plt.show()
    print(mean)

    print(time.time() - start)
    # Close environment after training is completed.
    env.close()
