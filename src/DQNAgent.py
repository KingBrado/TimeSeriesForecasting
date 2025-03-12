from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from src.DQN import DQN


# ======= Define the DQN Agent =======
class DQNAgent:
    def __init__(
            self,
            state_size,
            action_size,
            gamma = 0.99,
            epsilon = 1,
            epsilon_min = 0.01,
            epsilon_decay = 0.995,
            learning_rate = 0.001,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = 32

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[action] = torch.tensor(target, device=self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce exploration

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())  # Sync networks