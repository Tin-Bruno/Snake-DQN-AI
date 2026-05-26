import random

import numpy as np
import torch
from torch import nn, optim

from config import (
    ACTION_SIZE,
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LR,
    MEMORY_SIZE,
    MIN_MEMORY_SIZE,
    STATE_SIZE,
)
from memory.replay_buffer import ReplayBuffer
from models.dqn import DQN


class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(
            input_size=self.state_size,
            output_size=self.action_size,
        ).to(self.device)

        self.target_net = DQN(
            input_size=self.state_size,
            output_size=self.action_size,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.epsilon = EPSILON_START

    def choose_action(self, state):
        """
        Epsilon-greedy:
        - com chance epsilon: ação aleatória
        - senão: ação com maior Q-value
        """

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values).item())

    def train_step(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return None

        batch = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", EPSILON_END)