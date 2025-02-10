import os
import torch
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.TradingEnv import TradingEnv
from src.DQNAgent import DQNAgent

def load_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data["Adj Close"].values
    return prices


def save_memory(memory, filename):
    pd.DataFrame(memory, columns=['state', 'action', 'reward', 'next_state', 'done']).to_csv(filename)
   
# ======= Run Training =======
def train(train_data, model_dir, data_dir):
    env = TradingEnv(train_data)
    n_past_prices = 5
    state_size = n_past_prices + 1
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_size, action_size)

    num_episodes = 100
    for e in range(num_episodes):
        state = env.reset()
        if e % 10 == 0:
            portfolio_value = [(
                env.balance,
                env.holdings,
                env.balance + (env.holdings * env.data[-1]),
                None, None)]
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
        if e % 10 == 0:
            previous_value = portfolio_value[-1][2]
            current_value = env.balance + (env.holdings * env.data[-1])
            portfolio_value.append((
                env.balance,
                env.holdings,
                env.balance + (env.holdings * env.data[-1]),
                (current_value - previous_value) / previous_value,
                (env.data[-1]-env.data[0])/env.data[0]))
            pd.DataFrame(portfolio_value, columns=['balance', 'holdings', "portfolio", "return", "buy_and_hold"]).to_csv(f"{data_dir}/portfolio_value_{e}.csv")
        save_memory(agent.memory, f"{data_dir}/train_memory.csv")
        agent.replay()  # Train the model

    # Save the trained model
    model_name = f"{model_dir}/dqn_trading_agent_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth"
    torch.save(agent.model.state_dict(), model_name)
    print("Training complete. Model saved.")
    return model_name

def eval(eval_data, model_name, data_dir):
    env = TradingEnv(eval_data)
    n_past_prices = 5
    state_size = n_past_prices + 1
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(f"{model_name}"))
    agent.epsilon = 0
    state = env.reset()
    portfolio_value = [(
        env.balance,
        env.holdings,
        env.balance + (env.holdings * env.data[-1]),
        None, None
    )]
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        if done:
            break
    save_memory(agent.memory, f"{data_dir}/eval_memory.csv")
    previous_value = portfolio_value[-1][2]
    current_value = env.balance + (env.holdings * env.data[-1])
    portfolio_value.append((
        env.balance,
        env.holdings,
        env.balance + (env.holdings * env.data[-1]),
        (current_value - previous_value) / previous_value,
        (env.data[-1]-env.data[0])/env.data[0]
    ))
    pd.DataFrame(portfolio_value, columns=['balance', 'holdings', "portfolio", "return", "buy_and_hold"]).to_csv(f"{data_dir}/portfolio_value_eval.csv")


if __name__ == '__main__':
    for ticker in ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK-B', 'META', 'UNH']:
        os.makedirs(f"models/{ticker}", exist_ok=True)
        os.makedirs(f"{ticker}", exist_ok=True)
        
        prices = load_prices(ticker, start_date='2024-02-01', end_date='2025-02-01')
        mid_point = len(prices) // 2
        train_data = prices[:mid_point]
        model_name = train(train_data, model_dir=f"models/{ticker}", data_dir=f"{ticker}")

        eval_data = prices[mid_point:]
        eval(eval_data, model_name, data_dir=f"{ticker}")