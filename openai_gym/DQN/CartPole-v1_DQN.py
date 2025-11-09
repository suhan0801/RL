import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 創建 CartPole 環境
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # 4 個狀態變數
action_dim = env.action_space.n  # 2 個動作（左、右）

# 建立 DQN 模型
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # 第一層
            nn.ReLU(),
            nn.Linear(64, 64),  # 第二層
            nn.ReLU(),
            nn.Linear(64, action_dim)  # 輸出層 (Q 值)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化網路
policy_net = DQN(state_dim, action_dim)  # 主網路
target_net = DQN(state_dim, action_dim)  # 目標網路（每隔幾步更新）
target_net.load_state_dict(policy_net.state_dict())  # 初始化為相同權重
target_net.eval()  # 目標網路不進行梯度更新
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net.to(device)
target_net.to(device)

# 設定超參數
lr = 0.001  # 學習率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995  # 探索率衰減
min_epsilon = 0.01  # 最小探索率
batch_size = 64  # 訓練時取樣的批次大小
memory_size = 10000  # 經驗回放的記憶體大小
update_target_steps = 100  # 每 100 步更新目標網路
episodes = 1000  # 訓練回合數
optimizer = optim.Adam(policy_net.parameters(), lr=lr)  # Adam 優化器
loss_fn = nn.MSELoss()  # MSE 損失函數
replay_buffer = deque(maxlen=memory_size)  # 經驗回放記憶體

# 取得動作（ε-貪婪策略）
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # 隨機動作
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 轉換為 Tensor
        with torch.no_grad():
            q_values = policy_net(state_tensor)  # 預測 Q 值
        return torch.argmax(q_values).item()  # 選擇最大 Q 值的動作

# 訓練 DQN
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)  # 取得動作
        next_state, reward, done, truncated, _ = env.step(action)  # 執行動作
        replay_buffer.append((state, action, reward, next_state, done))  # 儲存經驗
        state = next_state
        total_reward += reward

        # 訓練網路
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)  # 隨機取樣
            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

            # 計算 Q 值
            q_values = policy_net(states_tensor).gather(1, actions_tensor)  # 選擇執行的動作對應的 Q 值

            # 計算目標 Q 值
            with torch.no_grad():
                max_next_q_values = target_net(next_states_tensor).max(1, keepdim=True)[0]
                target_q_values = rewards_tensor + gamma * max_next_q_values * (1 - dones_tensor)

            # 計算損失函數
            loss = loss_fn(q_values, target_q_values)

            # 更新網路
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 更新探索率
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # 每 100 步更新一次目標網路
    if episode % update_target_steps == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 每 10 回合輸出結果
    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}")

# 儲存模型
torch.save(policy_net.state_dict(), "dqn_cartpole.pth")

# 測試模型
state, _ = env.reset()
done = False
while not done:
    env.render()
    action = select_action(state, 0)  # 只用 Q 值選擇動作
    state, _, done, _, _ = env.step(action)

env.close()
