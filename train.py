"""
Noisy Dueling DQN for DemonAttack
- 无 epsilon-greedy，探索由网络噪声自动实现
- 包含奖励重塑（生存奖励、击杀放大、生命损失惩罚）
- 兼容 PyTorch 2.6+ 和 Gymnasium
"""

import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# ---------- 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 超参数 ----------
BUFFER_SIZE = 200000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0001
TARGET_UPDATE = 2000
LEARNING_STARTS = 10000
TRAIN_FREQ = 4
SAVE_INTERVAL = 100
FRAME_STACK = 4
ACTION_SIZE = 6

# NoisyNet 参数
NOISY_STD = 0.5          # 初始噪声标准差

# 奖励重塑参数
SURVIVAL_REWARD = 0.01
KILL_BONUS = 2.0
LIFE_LOSS_PENALTY = -2.0

# ---------- Noisy Linear 层 ----------
class NoisyLinear(nn.Module):
    """带参数化噪声的线性层，用于自动探索"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 可学习参数：均值
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # 可学习参数：标准差（对数空间，确保正值）
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # 注册噪声缓冲区（不参与梯度）
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # 使用常见的初始化方案
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # 标准差初始化为固定值（通过 softplus 保证为正）
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """重新生成噪声（每个 episode 或每步推荐重置）"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        # 使用因子化高斯噪声（使计算更高效）
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        # 采样时的权重 = 均值 + 标准差 * epsilon
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ---------- Noisy Dueling DQN 网络 ----------
class NoisyDuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # 卷积部分（无噪声）
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.feature_dim = 7 * 7 * 64

        # 价值流和优势流：使用 NoisyLinear
        self.value_fc = NoisyLinear(self.feature_dim, 512, NOISY_STD)
        self.value = NoisyLinear(512, 1, NOISY_STD)

        self.adv_fc = NoisyLinear(self.feature_dim, 512, NOISY_STD)
        self.adv = NoisyLinear(512, num_actions, NOISY_STD)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        val = F.relu(self.value_fc(x))
        val = self.value(val)

        adv = F.relu(self.adv_fc(x))
        adv = self.adv(adv)

        # Dueling 聚合
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        """重置网络中所有 NoisyLinear 层的噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ---------- 奖励重塑包装器（正确获取生命） ----------
class RewardShapingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 穿透包装器获取真正的 ALE 对象
        unwrapped = env.unwrapped
        while not hasattr(unwrapped, 'ale') and hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        self.ale = getattr(unwrapped, 'ale', None)
        self.last_lives = None

    def _get_lives(self):
        if self.ale is not None:
            return self.ale.lives()
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = self._get_lives()
        if self.last_lives is None:
            self.last_lives = info.get('lives', info.get('ale.lives', 3))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self._get_lives()
        if lives is None:
            lives = info.get('lives', info.get('ale.lives', self.last_lives))

        shaped_reward = reward  # 原始得分

        # 存活奖励
        shaped_reward += SURVIVAL_REWARD

        # 击杀放大
        if reward > 0:
            shaped_reward += KILL_BONUS

        # 生命损失惩罚
        if self.last_lives is not None and lives < self.last_lives:
            shaped_reward += LIFE_LOSS_PENALTY

        self.last_lives = lives
        return obs, shaped_reward, terminated, truncated, info

# ---------- 环境创建 ----------
def create_env(render_mode=None):
    # 必须禁用原始帧跳过
    env = gym.make("ALE/DemonAttack-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=4, terminal_on_life_loss=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = RewardShapingWrapper(env)
    return env

# ---------- 经验回放（简单均匀采样） ----------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.uint8),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.uint8),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ---------- DQN Agent（Noisy Dueling DQN） ----------
class NoisyDQNAgent:
    def __init__(self, action_size, load_checkpoint=None):
        self.action_size = action_size
        self.device = device

        self.policy_net = NoisyDuelingDQN(action_size).to(device)
        self.target_net = NoisyDuelingDQN(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        self.memory = ReplayBuffer(BUFFER_SIZE)

        self.steps_done = 0
        self.episode = 0

        if load_checkpoint and os.path.exists(load_checkpoint):
            self.load_checkpoint(load_checkpoint)

    def select_action(self, state):
        """NoisyNet 不需要 epsilon，网络自身产生随机性"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    q_values = self.policy_net(state_t)
            else:
                q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE or len(self.memory) < LEARNING_STARTS:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 混合精度前向
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                current_q = self.policy_net(states).gather(1, actions)
                with torch.no_grad():
                    # Double DQN: 策略网选动作，目标网评价值
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions)
                    target_q = rewards + GAMMA * next_q * (1 - dones)
                loss = F.mse_loss(current_q, target_q)
        else:
            current_q = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + GAMMA * next_q * (1 - dones)
            loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self):
        """在每个 episode 开始前重置策略网络的噪声"""
        self.policy_net.reset_noise()
        # 目标网络不需要重置噪声，因为它不参与探索

    def save_checkpoint(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode': self.episode,
        }, path)
        print(f"检查点保存至 {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps_done = ckpt['steps_done']
        self.episode = ckpt['episode']
        print(f"加载 {path}，继续 episode {self.episode}")

# ---------- 训练 ----------
def train(total_episodes, resume=False):
    ckpt_path = "noisy_demonattack.pth"
    env = create_env()
    agent = NoisyDQNAgent(ACTION_SIZE, load_checkpoint=ckpt_path if resume else None)

    start_ep = agent.episode + 1 if resume else 1
    all_rewards = []
    all_losses = []

    print(f"开始训练 Noisy Dueling DQN，共 {total_episodes} episodes")
    for ep in range(start_ep, total_episodes + 1):
        # 重置网络噪声（每个 episode 开始时重新采样噪声）
        agent.reset_noise()

        state, _ = env.reset()
        state = np.array(state, dtype=np.uint8)
        ep_reward = 0
        ep_losses = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.uint8)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.steps_done += 1

            if agent.steps_done >= LEARNING_STARTS and agent.steps_done % TRAIN_FREQ == 0:
                loss = agent.learn()
                if loss:
                    ep_losses.append(loss)

            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_network()

        all_rewards.append(ep_reward)
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        all_losses.append(avg_loss)
        agent.episode = ep

        if ep % 50 == 0:
            avg_r = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else ep_reward
            print(f"Ep {ep:5d} | Reward: {ep_reward:7.1f} | Avg50: {avg_r:7.1f} | Loss: {avg_loss:.4f} | Steps: {agent.steps_done}")

        if ep % SAVE_INTERVAL == 0:
            agent.save_checkpoint(ckpt_path)

    agent.save_checkpoint(ckpt_path)
    env.close()

    # 绘图
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(all_rewards)
        plt.title('Episode Rewards (Noisy Dueling DQN)')
        plt.xlabel('Episode')
        plt.subplot(1,2,2)
        plt.plot(all_losses)
        plt.title('Loss')
        plt.tight_layout()
        plt.savefig('noisy_rewards.png')
        print("训练曲线已保存为 noisy_rewards.png")
    except ImportError:
        pass

    return all_rewards, all_losses

# ---------- 评估 ----------
def evaluate(episodes=5, render=True, ckpt_path="noisy_demonattack.pth"):
    if not os.path.exists(ckpt_path):
        print(f"模型 {ckpt_path} 不存在，请先训练")
        return
    env = create_env(render_mode='human' if render else None)
    agent = NoisyDQNAgent(ACTION_SIZE, load_checkpoint=ckpt_path)
    agent.policy_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.uint8)
        total = 0
        done = False
        while not done:
            # 评估时网络无噪声（因为 eval 模式下 NoisyLinear 禁用噪声）
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            state = np.array(next_state, dtype=np.uint8)
            if render:
                time.sleep(0.02)
        print(f"演示回合 {ep+1}: 累积奖励 = {total:.1f}")
    env.close()

if __name__ == "__main__":
    #total_episodes为训练总回合，resume为是否继续训练
    #train(total_episodes=1500, resume=True)
    # episodes为演示轮数
    evaluate(episodes=3, render=True)