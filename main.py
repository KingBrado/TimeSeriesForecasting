import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from collections import deque
import yfinance as yf
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data["Adj Close"].values
    return prices

# ======= Define the DQN Model =======
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No activation (raw Q-values)
    

    # ======= Define the Trading Environment (Custom) =======
class TradingEnv:
    def __init__(self, data, n_last_prices=5, initial_balance=10000):
        if len(data) <= n_last_prices:
            raise ValueError("Data length must be greater than n_last_prices.")
        self.data = data
        self.n_last_prices = n_last_prices
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0
        self.last_action = 1
        self.old_portfolio_value = self.balance
        self.done = False

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0
        self.last_action = 1
        self.old_portfolio_value = self.balance
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            return None, 0, True

        price = self.data[self.current_step+self.n_last_prices]
        reward = 0

        # Execute action
        if action == 0:
            if self.balance >= price:
                self.holdings += 1
                self.balance -= price
        elif action == 1:
            pass
        elif action == 2:
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += price

        # Compute Reward (Profit Change - Trading Cost Penalty)
        new_portfolio_value = self.balance + (self.holdings * price)
        reward = new_portfolio_value - self.old_portfolio_value - abs(action - self.last_action) * 0.01
        self.last_action = action
        self.old_portfolio_value = new_portfolio_value

        # Advance time step
        self.current_step += 1
        if self.current_step+self.n_last_prices+1 >= len(self.data):
            self.done = True

        return self._get_state(), reward, self.done

    def _get_state(self):
        state = np.array(self.data[self.current_step:self.current_step+self.n_last_prices+1])
        return state
    

# ======= Define the DQN Agent =======
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.8
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state = torch.FloatTensor(state).to(device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce exploration

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())  # Sync networks


# ======= Run Training =======
def train(train_data):
    env = TradingEnv(train_data)
    n_past_prices = 5
    state_size = n_past_prices + 1
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_size, action_size)

    num_episodes = 100
    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_model()
                print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
                break

        agent.replay()  # Train the model

    # Save the trained model
    torch.save(agent.model.state_dict(), f"models/dqn_trading_agent_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth")
    print("Training complete. Model saved.")

if __name__ == '__main__':
    prices = load_prices('NVDA', start_date='2024-02-01', end_date='2025-02-01')
    mid_point = len(prices) // 2
    train_data = prices[:mid_point]
    train(train_data)