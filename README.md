# 強化學習專案集：CartPole 與 Unity RL

## 專案介紹
這個專案包含兩個主要部分：

###  CartPole Reinforcement Learning
使用 **OpenAI Gym** 的 **CartPole-v1** 環境，展示兩種強化學習方法：

- **Q-table**：利用表格方式紀錄每個離散化狀態下的動作價值 (Q-value)，適合簡單、離散狀態空間。
- **DQN (Deep Q-Network)**：使用神經網路來逼近 Q-value，適合連續狀態或高維環境。

目標是讓智能體學會控制小車保持桿子直立，並比較 Q-table 與 DQN 在學習效率與表現上的差異。

###  Unity Reinforcement Learning
- 使用 **Unity ML-Agents** 框架進行智能體訓練。
- 目前專案希望使用 **純視覺輸入（影像）** 作為狀態，搭配 **CNN + PPO** 演算法進行訓練。
- 目前因硬體設備限制（GPU 計算能力不足）尚未完成完整訓練。
- 專案未來目標：
  - 訓練智能體直接從遊戲畫面中學習決策
  - 探索 CNN 特徵抽取對 PPO 訓練穩定性與收斂速度的影響

Q-table

核心概念：

使用表格紀錄狀態-動作對的價值 Q(s, a)

更新公式（Q-learning）：

Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]


α: 學習率 (Learning Rate)

γ: 折扣因子 (Discount Factor)

r: 即時獎勵

優點：簡單、直觀，容易實作

缺點：狀態空間大或連續時不適用

DQN (Deep Q-Network)

核心概念：

使用神經網路逼近 Q(s,a)

搭配 經驗回放（Replay Buffer），隨機抽取 mini-batch 更新，避免樣本相關性問題

使用 目標網路（Target Network） 增穩，減少 Q 值震盪

更新公式：

Loss = (Q_target - Q(s,a))^2
Q_target = r + γ * max_a' Q_target(s',a')


進階技巧：

Double DQN：解決 DQN 過高估計 Q 值問題

Dueling DQN：將 state value 與 advantage 分開，提高學習效率

ε-greedy 探索策略：平衡探索與利用

Unity RL

Unity RL (CNN + PPO)

核心概念：

PPO (Proximal Policy Optimization)：策略梯度演算法，適合連續或離散動作空間，穩定性高

CNN (Convolutional Neural Network)：從遊戲畫面（影像）抽取特徵，作為 PPO 的狀態輸入

技術細節：

將 Unity 畫面轉成灰階或 RGB 影像，進行 resize、normalize

CNN 提取空間特徵後輸入策略網路 (Policy Network)

搭配 Reward Shaping 設計適合的獎勵函數

可使用 Curriculum Learning 漸進式提升環境難度

現況：

目前因硬體設備限制尚未完成完整訓練

未來將補上 CNN + PPO 模型、訓練曲線與性能分析

使用 Curriculum Learning 漸進式訓練環境難度

現況：因硬體限制尚未完成完整訓練，未來將補上完整模型與訓練曲線
