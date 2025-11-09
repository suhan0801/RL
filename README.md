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
- 目前專案因硬體設備限制（GPU 性能不足）尚未完成完整訓練。
- 專案未來將使用 **DQN、Double DQN、PPO (Proximal Policy Optimization)** 等演算法控制 Unity 環境中的角色或物理物件。

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

使用框架：Unity ML-Agents

適用演算法：

DQN / Double DQN：適合離散動作空間

PPO (Proximal Policy Optimization)：適合連續動作空間，穩定性高

核心技術：

將 Unity 環境封裝成 Gym-like API

使用 Reward Shaping 設計適合的獎勵函數

使用 Curriculum Learning 漸進式訓練環境難度

現況：因硬體限制尚未完成完整訓練，未來將補上完整模型與訓練曲線
