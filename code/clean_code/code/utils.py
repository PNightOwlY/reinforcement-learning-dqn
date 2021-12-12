import random

import gym
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')


def preprocessPong(state, next_state):
  # add one frame, and remove the first frame
  state = state[1:]
  state.append(next_state)

  # change to float type and vary the range from (1,255) to (0,1)
  state_tensor = torch.tensor(state).float().unsqueeze(0) / 255

  # put the state to device
  return state_tensor.to(device), state
