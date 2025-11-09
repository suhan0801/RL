import gym
import numpy as np

# 創建環境，啟用 human 模式來顯示畫面
env = gym.make("CartPole-v1", render_mode="human")

# 讀取訓練好的 Q 表格（這裡假設已經存成 numpy 檔案）
Q = np.load("q_table.npy")  # 確保這個檔案存在

# 定義狀態離散化函數（要跟訓練時的一致）
state_bins = [np.linspace(-2.4, 2.4, 10),
              np.linspace(-2, 2, 10),
              np.linspace(-0.209, 0.209, 10),
              np.linspace(-2, 2, 10)]

def discretize_state(state):
    return tuple(np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state)))

# 測試 AI
state, _ = env.reset()
state = discretize_state(state)
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])  # 選擇最佳動作
    next_state, reward, done, truncated, _ = env.step(action)
    state = discretize_state(next_state)
    total_reward += reward

print(f"Total Reward: {total_reward}")


env.close()
