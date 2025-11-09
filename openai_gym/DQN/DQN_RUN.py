import gym
import torch
import torch.nn as nn

# 創建 CartPole 環境
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]  # 4 個狀態變數
action_dim = env.action_space.n  # 2 個動作（左、右）

# **建立 DQN 模型（需與訓練時相同）**
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# **載入模型**
policy_net = DQN(state_dim, action_dim)
policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))  # 載入權重
policy_net.eval()  # 設為評估模式，不更新梯度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net.to(device)



# **測試模型**
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # 顯示畫面
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 轉換為 Tensor
    with torch.no_grad():
        action = torch.argmax(policy_net(state_tensor)).item()  # 選擇 Q 值最大的動作
    state, reward, done, _, _ = env.step(action)  # 執行動作
    total_reward += reward

print(f"Total Reward: {total_reward}")
env.close()
