import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.batch_size_win = env_config["batch_size_win"]
        self.threshold = env_config["threshold"]
        self.step = 0
        self.eps = self.eps_start


        self.conv1 = nn.Conv2d(4, 32, 8, stride= 4)
        self.conv1.weight.data.normal_(0,0.1)

        self.conv2 = nn.Conv2d(32, 64, 4, stride= 2)
        self.conv2.weight.data.normal_(0, 0.1)

        self.conv3 = nn.Conv2d(64, 64, 3, stride= 1)
        self.conv3.weight.data.normal_(0, 0.1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(512, self.n_actions)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        action_value = self.fc2(x)

        return action_value

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        # self.eps -= (self.eps_start - self.eps_end) / self.anneal_length
        # if self.eps < self.eps_end:
        #   self.eps = self.eps_end

        if np.random.uniform() < 1 - self.eps:  # greedy
          actions_value = self.forward(observation)
          action = torch.max(actions_value, 1)[1].cpu().data.numpy()
          action = action[0]
        else:  # random
          action = np.random.randint(0, self.n_actions)
          action = action
        return action
        raise NotImplementedError

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.threshold:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    sample = memory.sample(dqn.batch_size)

    batch_s = torch.stack(sample[0]).view(dqn.batch_size, 4, 84, 84)
    batch_a = torch.stack(sample[1]).view(dqn.batch_size, 1)
    batch_s_ = torch.stack(sample[2]).view(dqn.batch_size, 4, 84, 84)
    batch_r = torch.stack(sample[3]).view(dqn.batch_size, 1)
    batch_done = torch.stack(sample[4]).view(dqn.batch_size, 1)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn(batch_s).gather(1, batch_a)
    q_next = target_dqn(batch_s_).detach()
    q_value_targets = batch_r + dqn.gamma * q_next.max(1)[0].view(dqn.batch_size,1) * batch_done

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

    # Compute loss.
    loss = F.mse_loss(q_values, q_value_targets)
    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()


def optimize_win(dqn, target_dqn, memory, optimizer):
  """This function samples a batch from the replay buffer and optimizes the Q-network."""
  # If we don't have enough transitions stored yet, we don't train.
  if len(memory) < 5:
    return

  # TODO: Sample a batch from the replay memory and concatenate so that there are
  #       four tensors in total: observations, actions, next observations and rewards.
  #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
  #       Note that special care is needed for terminal transitions!

  batch = 5

  sample = memory.sample(batch)

  batch_s = torch.stack(sample[0]).view(batch, 4, 84, 84)
  batch_a = torch.stack(sample[1]).view(batch, 1)
  batch_s_ = torch.stack(sample[2]).view(batch, 4, 84, 84)
  batch_r = torch.stack(sample[3]).view(batch, 1)
  batch_done = torch.stack(sample[4]).view(batch, 1)

  # TODO: Compute the current estimates of the Q-values for each state-action
  #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
  #       corresponding to the chosen actions.
  q_values = dqn(batch_s).gather(1, batch_a)
  q_next = target_dqn(batch_s_).detach()
  q_value_targets = batch_r + dqn.gamma * q_next.max(1)[0].view(batch, 1) * batch_done

  # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

  # Compute loss.
  loss = F.mse_loss(q_values, q_value_targets)
  # Perform gradient descent.
  optimizer.zero_grad()

  loss.backward()
  optimizer.step()

  return loss.item()
