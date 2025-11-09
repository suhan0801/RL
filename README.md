# CartPole Reinforcement Learning: Q-table & DQN

## 專案介紹
這個專案使用 **OpenAI Gym** 的 **CartPole-v1** 環境，展示兩種強化學習方法的應用：

1. **Q-table**：利用表格方式紀錄每個離散化狀態下的動作價值 (Q-value)，適合簡單、離散狀態空間。
2. **DQN (Deep Q-Network)**：使用神經網路來逼近 Q-value，適合連續狀態或高維環境。

專案目標是讓智能體學會控制小車保持桿子直立，並比較 Q-table 與 DQN 在學習效率與表現上的差異。

---

核心概念
Q-table

適用：狀態離散化後的小型環境


缺點：狀態數量一多就無法擴展（維度災難）

DQN

適用：連續或高維狀態空間

優點：能處理大型環境，學習穩定

技巧：

使用經驗回放（Replay Buffer）

使用目標網路（Target Network）

搭配 ε-greedy 探索策略

結果展示

Q-table 收斂速度較慢，但對簡單環境有效。

DQN 能穩定收斂到高分，並能處理更多環境變化。
