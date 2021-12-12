import argparse
import random
import gym
import torch
import torch.nn as nn
from utils import preprocessPong

import config
from utils import preprocess
from gym.wrappers import AtariPreprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'],default='Pong-v0')
parser.add_argument('--path', type=str, help= f'models/Pong-v0_best.pt.', default=f'models/CartPole-v0_best.pt')
parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}

def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0

    for i in range(n_episodes):

        state = env.reset()

        obss = [state for _ in range(4)]

        obs, obss = preprocessPong(obss, state)

        done = False
        episode_return = 0
        while not done:

            env.render()
            action = dqn.act(obs, exploit=True)

            # Act in the true environment.
            next_state, reward, done, info = env.step(action + 2)

            obs, obss = preprocessPong(obss, next_state)

            episode_return += reward
        total_return += episode_return

        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')


    return total_return / n_episodes

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    # Initialize environment and config
    env = gym.make(args.env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]

    if args.save_video:
        env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()
