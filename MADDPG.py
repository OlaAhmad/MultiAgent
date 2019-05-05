"""
This a version of code trains two agents to play tennis using DDPG with multi-agents.
All agents share the same actor-critic network -- A self-play mode
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # tau for soft update of target parameters
LR_ACTOR = 1e-4             # learning rate of the actor
LR_CRITIC = 1e-3            # learning rate of the critic
WEIGHT_DECAY = 0            # L2 weight decay
NUM_AGENTS = 2
UPDATE_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    """
    Agent class to interact with the environment and learn from it.
    """
    def __init__(self, state_size, action_size, random_seed):
        """
        Initialize Agnet class.

        :param state_size: dimension of the state
        :param action_size: dimension of the action
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.seed = random.seed(random_seed)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        #
        self.noise = OUNoise(action_size, random_seed)

        # replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """
        save experience in memory.

        :param state: real value
        :param action: real value
        :param reward: real value
        :param next_state: real value
        :param done: boolean
        :return:
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):

            self.memory.add(state, action, reward, next_state, done)
            # Learn, if there are enough samples in memory
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, states, add_noise=True):
        """
        return actions for given state and current policy.

        :param state: real value
        :param add_noise: boolean
        :return: action value
        """
        actions = np.zeros((NUM_AGENTS, self.action_size))
        for i in range(NUM_AGENTS):
            state = states[i]
            state = torch.from_numpy(state).float().to(device)
            self.actor_local.eval()

            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()

            self.actor_local.train()
            if add_noise:
                action += self.noise.sample()

            actions[i,...] = action
        return np.clip(actions, -1, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        update policy.

        :param experience: tuple of enviornment values
        :param gamma: discount factor
        :return:
        """
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        soft update model's parameters.

        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1-tau) * target_params.data)

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.5, sigma_min=0.05, sigma_decay=0.99):
        """
        initialize noise process.

        :param size: dimensions of the action
        :param seed: random seed
        :param mu: 1st parameter
        :param theta: 2nd pqrqmeter
        :param sigma: 3rd parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        reset the internal state to mean.
        :return:
        """
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_decay * self.sigma, self.sigma_min)

    def sample(self):
        """
        update the internal state and return it as a noise process.
        :return:
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state


class ReplayBuffer:
    """
    Fixed size buffer to save experience tuples.
    """
    def __init__(self, action_size, buffer_size, batch_size, random_seed):
        """
        initialize class.

        :param action_size: dimension of the action
        :param buffer_size: size of memory
        :param batch_size: minibatch size
        :param random_seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        """
        add a new experience to the buffer
        :return:
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        randomly sample a batch of experiences from the memory
        :return:
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        return the current size of internal memory.
        :return:
        """
        return len(self.memory)




