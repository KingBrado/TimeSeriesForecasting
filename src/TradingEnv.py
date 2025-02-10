import numpy as np

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