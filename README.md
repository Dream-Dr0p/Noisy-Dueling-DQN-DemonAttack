# Noisy Dueling DQN for DemonAttack 🎮

基于 **Noisy Dueling Deep Q-Network** 的 Atari 游戏强化学习智能体，在 [DemonAttack（恶魔攻击）](https://gymnasium.farama.org/environments/atari/demon_attack/) 环境中训练，使用 PyTorch + Gymnasium/ALE 实现。

## 🧠 算法概述

本项目融合了 DQN 的三个核心改进：

| 组件 | 说明 |
|---|---|
| **Dueling Network** | 将 Q 网络拆分为 Value 流和 Advantage 流，分离"状态价值"与"动作优势" |
| **NoisyNet** | 用可学习的参数化噪声替代 epsilon-greedy，实现状态依赖的自动探索 |
| **Double DQN** | 策略网络选动作 + 目标网络评价值，缓解 Q 值过分估计偏差 |

此外还加入了 **奖励重塑（Reward Shaping）**：生存奖励 + 击杀放大 + 生命损失惩罚，加速智能体收敛。

## 📁 项目结构

```
.
├── train.py                          # 核心代码：Noisy Dueling DQN 实现与训练/评估
├── train.txt                         # 1500 局完整训练日志
├── DemonAttack.docx                  # 游戏参考文档
├── gymnasium.docx                    # Gymnasium 框架参考文档
├── DemonAttack.mp4                   # 1500 局训练后一般视频
└── README.md
```

## 📊 训练结果

训练 **1500 episodes**，共约 **180 万步**，使用 NVIDIA GPU + CUDA 混合精度加速。

| 指标 | 数值 |
|---|---|
| 历史最高单局得分 | **1643.0**（第 950 局） |
| 最终 Avg50（近50局均值） | **553.9**（第 1500 局） |
| 最终探索步数 | ~1,809,202 |
| 终点 Avg50 峰值 | **843.1**（第 1400 局附近） |

训练曲线（生成自 `train.py`）：

![训练曲线](noisy_rewards.png)

> 智能体从最初 ~90 分的随机水平逐步学会识别并击杀敌人，最终稳定在高分段。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch ≥ 2.0（支持 CUDA 可选，CPU 也可运行）
- Gymnasium + ALE（Atari 环境）

### 安装依赖

```bash
pip install torch gymnasium ale-py matplotlib numpy
pip install "gymnasium[atari]" "gymnasium[accept-rom-license]"
```

### 训练

```bash
# 从头训练 1500 局
python train.py
```

修改 `train.py` 底部代码可调整：

```python
if __name__ == "__main__":
    train(total_episodes=1500)          # 从头训练
    # train(total_episodes=1500, resume=True)  # 从检查点继续
```

### 评估 / 演示

```bash
# 使用训练好的模型进行游戏演示
python train.py  # evaluate() 默认运行 3 局
```

也支持直接调用：

```python
evaluate(episodes=5, render=True, ckpt_path="noisy_demonattack.pth")
```

### 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `BUFFER_SIZE` | 200,000 | 经验回放缓冲区大小 |
| `BATCH_SIZE` | 64 | 训练批次大小 |
| `GAMMA` | 0.99 | 折扣因子 |
| `LR` | 0.0001 | 学习率 |
| `TARGET_UPDATE` | 2,000 | 目标网络更新频率（步） |
| `NOISY_STD` | 0.5 | 初始噪声标准差 |
| `FRAME_STACK` | 4 | 输入帧堆叠数 |
| `LEARNING_STARTS` | 10,000 | 预热步数 |

## 🔧 技术特点

- **混合精度训练**：使用 `torch.cuda.amp` + `GradScaler` 在 GPU 上加速训练
- **因子化高斯噪声**：`NoisyLinear` 层实现高效的因子化噪声采样
- **奖励重塑包装器**：穿透 ALE 获取准确生命值，实现击杀奖励放大和死亡惩罚
- **检查点自动保存**：每 100 局自动保存模型权重，支持断点续训
- **训练曲线可视化**：自动生成 reward/loss 折线图

## 📄 License

MIT License

---

*本项目为大学综合实验设计课程作品，欢迎交流学习。*
