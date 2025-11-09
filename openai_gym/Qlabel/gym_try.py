import gym
import numpy as np
import random

# 創建 CartPole 環境
env = gym.make("CartPole-v1")

# 設定 Q 表格（觀察空間要離散化）
state_bins = [np.linspace(-2.4, 2.4, 10),  # 車的位置，範圍 [-2.4, 2.4]，分成 10 個區間
              np.linspace(-2, 2, 10),      # 車的速度，範圍 [-2, 2]，分成 10 個區間
              np.linspace(-0.209, 0.209, 10),  # 杆子角度，範圍 [-0.209, 0.209]，分成 10 個區間
              np.linspace(-2, 2, 10)]  # 杆子的角速度，範圍 [-2, 2]，分成 10 個區間

# 建立 Q 表格，四個維度代表觀察狀態的離散索引，第五個維度代表可執行的動作（左、右）
Q = np.zeros([10, 10, 10, 10, env.action_space.n])  

# 設定參數
alpha = 0.1  # 學習率，控制 Q 值更新的速度
gamma = 0.99  # 折扣因子，決定未來獎勵對當前 Q 值的影響程度
epsilon = 1.0  # 探索率，初始時完全隨機選擇動作
epsilon_decay = 0.995  # 探索率衰減，每回合逐漸減少隨機選擇的機率
min_epsilon = 0.01  # 最小探索率，確保探索不會完全停止
episodes = 10000  # 訓練回合數

# 轉換觀察值為離散狀態
def discretize_state(state):
    """將連續的狀態轉換為離散索引"""
    state_idx = tuple(np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state)))
    return state_idx

# 訓練 Q-learning
for episode in range(episodes):
    state = discretize_state(env.reset()[0])  # Gym 版本不同，reset 可能回傳 5 個值，取第 1 個
    done = False
    total_reward = 0  # 記錄當回合的總獎勵

    while not done:
        # 選擇動作（ε-貪婪策略）
        if random.uniform(0, 1) < epsilon:  
            action = env.action_space.sample()  # 隨機選擇動作（探索）
        else:
            action = np.argmax(Q[state])  # 選擇 Q 值最大的動作（利用）

        # 執行動作，取得新的狀態與獎勵
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)  # 離散化新狀態
        total_reward += reward  # 累積獎勵

        # Q-learning 更新 Q 值
        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]))

        # 轉移到下一個狀態
        state = next_state

    # 更新 ε 以減少隨機探索
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # 每 100 回合輸出一次訓練資訊
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# 測試訓練後的 Q-learning 模型
state = discretize_state(env.reset()[0])  # 重新初始化環境
done = False
while not done:
    env.render()  # 顯示畫面
    action = np.argmax(Q[state])  # 使用訓練好的 Q 表來選擇最佳動作
    next_state, reward, done, truncated, _ = env.step(action)  # 執行動作
    state = discretize_state(next_state)  # 轉換新狀態為離散索引

# 存儲 Q 表格，方便後續使用
np.save("q_table.npy", Q)  

# 關閉環境
env.close()
