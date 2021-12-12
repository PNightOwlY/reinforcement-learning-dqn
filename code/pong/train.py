import argparse
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import numpy as np

import config
from utils import preprocessPong
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize, optimize_win
import time
from PIL import Image
import math
from gym.wrappers import AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['Pong-v0'], default='Pong-v0')
parser.add_argument('--evaluate_freq', type=int, default=10000, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=3, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
  'Pong-v0':config.Pong
}
if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

  # Initialize environment and config.
    env = gym.make(args.env)
    # env = env.unwrapped
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # load the parameters for targer_dqn
    # dqn = torch.load("models/Pong-v0_best-2.pt", map_location=torch.device('cpu'))
    # target_dqn = torch.load("models/Pong-v0_best-2.pt", map_location=torch.device('cpu'))
    # dqn.batch_size = 32

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])
    winMemory = ReplayMemory(10000)
    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])
    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    best_reward = -float("Inf")

    epsilon_decay = lambda frame_idx: dqn.eps_end + (dqn.eps_start - dqn.eps_end) * math.exp(
      -1. * frame_idx / env_config["decay"])

    # plt.plot([epsilon_decay(i) for i in range(200000)])
    # plt.show()
    writer = SummaryWriter()

    rewardlist = []

    while(dqn.step < env_config['total_frames']):
    # for episode in range(env_config['n_episodes']):
        done = False

        # first observation contains n same frames
        state = env.reset()
        obss = [state for _ in range(env_config['n_frame'])]

        last_obs, obss = preprocessPong(obss, state)
        current_reward = 0
        step = 0

        while not done:
            # TODO: Get action from DQN.

            # the return value is 0,1 but the real value is [2,3]
            action = dqn.act(last_obs)
            dqn.eps = epsilon_decay(dqn.step)

            if dqn.eps < dqn.eps_end:
              dqn.eps = self.eps_end
            # env.render()
            # time.sleep(0.05)


            # Act in the true environment.
            next_state, reward, done, info = env.step(action+2)

            next_obs, obss = preprocessPong(obss, next_state)
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!

            if done:
              isdone = 0
            else:
              isdone = 1

            memory.push(last_obs, torch.tensor(action).to(device),
                        next_obs, torch.tensor(reward).to(device),
                        torch.tensor(isdone).to(device))



            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if dqn.step % env_config["train_frequency"] == 0:
              optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if dqn.step % env_config["target_update_frequency"] == 0:
              # optimize_win(dqn, target_dqn, winMemory, optimizer)
              target_dqn.load_state_dict(dqn.state_dict())

            if dqn.step % args.evaluate_freq == 0:
              # mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
              print("frames: %5d, mean_return: %5f, epsilon: %5f" % (dqn.step, np.mean(rewardlist[-10:]), dqn.eps))
              # writer.add_scalar("Mean Reward", np.mean(rewardlist[-10:], (int)(dqn.step/1000)))
              torch.save(dqn, f'models/{args.env}_best.pt')

              # if mean_return >= best_mean_return:
              #   best_mean_return = mean_return
              #   # print('Best performance so far! Saving model.')
              #   torch.save(dqn, f'models/{args.env}_best.pt')
            dqn.step = dqn.step + 1
            step += 1
            current_reward += reward
            last_obs = next_obs
        # To save time, the evaluation part is skipped, will do it when parameter tuning
        # Evaluate the current agent.
        # if episode % args.evaluate_freq == 0:
        #     mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
        #
        #     print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')
        #
        #     # Save current agent if it has the best performance so far.
        #     if mean_return >= best_mean_return:
        #         best_mean_return = mean_return
        #
        #         print('Best performance so far! Saving model.')
        #         torch.save(dqn, f'models/{args.env}_best.pt')

        rewardlist.append(current_reward)
    # Close environment after training is completed.
    env.close()
    plt.plot(range(1, env_config['total_frames']/1000+1), rewardlist)
    plt.show()
