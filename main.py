import os
import torch
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.TradingEnv import TradingEnv
from src.DQNAgent import DQNAgent

def load_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data["Close"].values
    return prices


def save_memory(memory, filename):
    pd.DataFrame(memory, columns=['state', 'action', 'reward', 'next_state', 'done']).to_csv(filename)
   
# ======= Run Training =======
def train(train_data, model_name, params):
    env = TradingEnv(train_data)
    n_past_prices = 5
    state_size = n_past_prices + 1
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_size, action_size, **params)

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
                print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward[0]:.2f}, Epsilon: {agent.epsilon:.4f}")
                break
        agent.replay()  # Train the model

    # Save the trained model
    torch.save(agent.model.state_dict(), model_name)
    print("Training complete. Model saved.")

def eval(eval_data, model_name):
    env = TradingEnv(eval_data)
    n_past_prices = 5
    state_size = n_past_prices + 1
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(f"{model_name}"))
    agent.epsilon = 0
    state = env.reset()
    initial_porfolio = env.balance + (env.holdings * env.data[0])
    while True:
        action = agent.act(state)
        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break
    final_portfolio = env.balance + (env.holdings * env.data[-1])
    return (final_portfolio - initial_porfolio) / initial_porfolio

if __name__ == '__main__':
    for ticker in ['BTC-USD']: #, 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK-B', 'META', 'UNH']:
        os.makedirs(f"models/{ticker}", exist_ok=True)
        os.makedirs(f"{ticker}", exist_ok=True)
        
        prices = load_prices(ticker, start_date='2024-02-01', end_date='2025-02-01')
        mid_point = len(prices) // 2
        train_data = prices[:mid_point]
        params = {
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001
        }
        model_name = f"models/{ticker}/dqn_trading_agent_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth"
        train(
            train_data,
            model_name=model_name,
            params=params
        )

        eval_data = prices[mid_point:]
        final_return = eval(eval_data, model_name)
        print(f"Final Portfolio Return: {final_return:.2f}")