"""
贪吃蛇 V6 Rust 加速版 - 8~10小时长训练版
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
import os

# --- 导入你的 Rust 引擎 ---
import fast_snake_v6

# --- 配置参数 ---
N_ENVS = 128

# 训练参数
MAX_MEMORY = 1_000_000
BATCH_SIZE = 4096
LR = 0.0005                 # 降低初始LR，配合高gamma更稳定
GAMMA = 0.99                # ★ 关键改动：大幅提升远见能力
GRID_SIZE = 20
N_ACTIONS = 3
N_SPATIAL_CHANNELS = 3
N_AUX_FEATURES = 21

# Epsilon
EPSILON_START = 1.0
EPSILON_MIN = 0.005          # ★ 接近0但保留微量探索
EPSILON_DECAY_STEPS = 180_000  # ★ 按训练步计，约前5-6小时完成衰减

MIN_REPLAY_SIZE = 50_000
TRAIN_EVERY_N_STEPS = 1
TARGET_TAU = 0.005

MAX_TOTAL_GAMES = 500_000    # ★ 大幅提升，让时间成为瓶颈

# 长训练专用：时间限制（秒）
MAX_TRAINING_TIME = 9 * 3600  # 9小时安全上限

# Scheduler
SCHEDULER_T_MAX = 300_000    # ★ 匹配预估训练步数


# --- 1. 网络架构 ---
class HybridDuelingQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_SPATIAL_CHANNELS, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, N_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE)
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            dummy = F.relu(self.conv3(dummy))
            self.cnn_feat_size = dummy.view(1, -1).size(1)

        self.aux_fc1 = nn.Linear(N_AUX_FEATURES, 128)
        self.aux_fc2 = nn.Linear(128, 64)
        combined_size = self.cnn_feat_size + 64

        self.val_fc1 = nn.Linear(combined_size, 256)
        self.val_ln = nn.LayerNorm(256)
        self.val_fc2 = nn.Linear(256, 128)
        self.val_out = nn.Linear(128, 1)

        self.adv_fc1 = nn.Linear(combined_size, 256)
        self.adv_ln = nn.LayerNorm(256)
        self.adv_fc2 = nn.Linear(256, 128)
        self.adv_out = nn.Linear(128, N_ACTIONS)

        self.dropout = nn.Dropout(0.1)

    def forward(self, spatial, aux):
        x = self.pool(F.relu(self.conv1(spatial)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        a = F.relu(self.aux_fc1(aux))
        a = F.relu(self.aux_fc2(a))

        combined = torch.cat([x, a], dim=1)
        combined = self.dropout(combined)

        v = F.relu(self.val_ln(self.val_fc1(combined)))
        v = F.relu(self.val_fc2(v))
        v = self.val_out(v)

        adv = F.relu(self.adv_ln(self.adv_fc1(combined)))
        adv = F.relu(self.adv_fc2(adv))
        adv = self.adv_out(adv)

        return v + adv - adv.mean(dim=1, keepdim=True)


# --- 2. 高效 Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.spatials = np.zeros((capacity, N_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.auxs = np.zeros((capacity, N_AUX_FEATURES), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_spatials = np.zeros((capacity, N_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.next_auxs = np.zeros((capacity, N_AUX_FEATURES), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push_batch(self, s, a, act, r, ns, na, d):
        n = len(s)
        remaining = self.capacity - self.ptr
        if n <= remaining:
            self.spatials[self.ptr:self.ptr + n] = s
            self.auxs[self.ptr:self.ptr + n] = a
            self.actions[self.ptr:self.ptr + n] = act
            self.rewards[self.ptr:self.ptr + n] = r
            self.next_spatials[self.ptr:self.ptr + n] = ns
            self.next_auxs[self.ptr:self.ptr + n] = na
            self.dones[self.ptr:self.ptr + n] = d
            self.ptr = (self.ptr + n) % self.capacity
        else:
            self.spatials[self.ptr:] = s[:remaining]
            self.auxs[self.ptr:] = a[:remaining]
            self.actions[self.ptr:] = act[:remaining]
            self.rewards[self.ptr:] = r[:remaining]
            self.next_spatials[self.ptr:] = ns[:remaining]
            self.next_auxs[self.ptr:] = na[:remaining]
            self.dones[self.ptr:] = d[:remaining]

            overflow = n - remaining
            self.spatials[:overflow] = s[remaining:]
            self.auxs[:overflow] = a[remaining:]
            self.actions[:overflow] = act[remaining:]
            self.rewards[:overflow] = r[remaining:]
            self.next_spatials[:overflow] = ns[remaining:]
            self.next_auxs[:overflow] = na[remaining:]
            self.dones[:overflow] = d[remaining:]
            self.ptr = overflow

        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.spatials[idx], self.auxs[idx], self.actions[idx],
            self.rewards[idx], self.next_spatials[idx], self.next_auxs[idx],
            self.dones[idx],
        )


# --- 3. Agent ---
class VectorDQNAgent:
    def __init__(self, device):
        self.device = device
        self.policy_net = HybridDuelingQNet().to(device)
        self.target_net = HybridDuelingQNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=SCHEDULER_T_MAX, eta_min=1e-5
        )
        self.memory = ReplayBuffer(MAX_MEMORY)
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def select_actions(self, spatials, auxs):
        n_envs = spatials.shape[0]
        actions = np.zeros(n_envs, dtype=np.int32)

        rand_mask = np.random.random(n_envs) < self.epsilon
        actions[rand_mask] = np.random.randint(0, N_ACTIONS, size=np.sum(rand_mask))

        net_indices = np.where(~rand_mask)[0]
        if len(net_indices) > 0:
            self.policy_net.eval()
            sp_t = torch.from_numpy(spatials[net_indices]).to(self.device)
            ax_t = torch.from_numpy(auxs[net_indices]).to(self.device)

            with torch.no_grad():
                q_vals = self.policy_net(sp_t, ax_t)
                net_acts = q_vals.argmax(dim=1).cpu().numpy()

            self.policy_net.train()
            actions[net_indices] = net_acts

        return actions.tolist()

    def train_step(self):
        if self.memory.size < MIN_REPLAY_SIZE:
            return None

        sp, ax, act, rew, nsp, nax, dn = self.memory.sample(BATCH_SIZE)

        sp_t = torch.from_numpy(sp).to(self.device)
        ax_t = torch.from_numpy(ax).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        nsp_t = torch.from_numpy(nsp).to(self.device)
        nax_t = torch.from_numpy(nax).to(self.device)
        dn_t = torch.from_numpy(dn).to(self.device)

        self.policy_net.train()
        curr_q = self.policy_net(sp_t, ax_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            self.policy_net.eval()
            next_acts = self.policy_net(nsp_t, nax_t).argmax(dim=1)
            self.policy_net.train()
            next_q = self.target_net(nsp_t, nax_t).gather(1, next_acts.unsqueeze(1)).squeeze(1)
            target_q = rew_t + GAMMA * next_q * (1 - dn_t)

        loss = F.smooth_l1_loss(curr_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.steps_done += 1

        # Soft target update
        with torch.no_grad():
            for pt, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                pt.data.mul_(1 - TARGET_TAU).add_(pp.data, alpha=TARGET_TAU)

        # Epsilon衰减
        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN,
                               self.epsilon - (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY_STEPS)

        return loss.item()

    def save(self, path):
        torch.save({
            'model': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps_done,
            'eps': self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(ckpt['model'])
        if 'target' in ckpt:
            self.target_net.load_state_dict(ckpt['target'])
        else:
            self.target_net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.steps_done = ckpt['steps']
        self.epsilon = ckpt['eps']


# --- 4. 主循环 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on {device} with {N_ENVS} Rust environments!")

    agent = VectorDQNAgent(device)

    env = fast_snake_v6.VectorizedSnakeEnv(N_ENVS)

    if os.path.exists("snake_v6_rust.pth"):
        agent.load("snake_v6_rust.pth")
        print("Loaded checkpoint.")

    obs_spatial, obs_aux = env.get_states_batch()

    total_games = 0
    scores_window = deque(maxlen=100)
    record = 0
    start_time = time.time()
    total_steps = 0

    print_timer = time.time()
    step_counter = 0
    recent_loss = deque(maxlen=200)

    # ★ 追踪最佳平均分，用于保存"平均表现最好"的模型
    best_avg_score = 0.0

    print(f"\nConfig: BATCH={BATCH_SIZE}, LR={LR}, GAMMA={GAMMA}, N_ENVS={N_ENVS}")
    print(f"Epsilon: {EPSILON_START} -> {EPSILON_MIN}, decay_steps={EPSILON_DECAY_STEPS}")
    print(f"Buffer: {MAX_MEMORY}, min_replay: {MIN_REPLAY_SIZE}")
    print(f"Target: max_games={MAX_TOTAL_GAMES}, max_time={MAX_TRAINING_TIME}s ({MAX_TRAINING_TIME/3600:.1f}h)")
    print(f"Scheduler T_max: {SCHEDULER_T_MAX}")
    print()

    while total_games < MAX_TOTAL_GAMES:
        # ★ 时间安全阀：超时则优雅退出
        elapsed = time.time() - start_time
        if elapsed > MAX_TRAINING_TIME:
            print(f"\n⏰ Time limit reached ({elapsed/3600:.1f}h). Stopping gracefully.")
            break

        # 1. 选择动作
        actions = agent.select_actions(obs_spatial, obs_aux)

        # 2. 环境步进
        rewards, dones, scores = env.step_batch(actions)

        # 3. 获取新状态
        next_spatial, next_aux = env.get_states_batch()

        # 4. 存入经验池
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        agent.memory.push_batch(
            obs_spatial, obs_aux, np.array(actions),
            rewards_np, next_spatial, next_aux, dones_np,
        )

        # 5. 更新当前状态
        obs_spatial, obs_aux = next_spatial, next_aux

        # 6. 统计分数
        for i in range(N_ENVS):
            if dones[i]:
                total_games += 1
                game_score = int(scores[i])
                scores_window.append(game_score)
                if game_score > record:
                    record = game_score
                    agent.save("snake_v6_rust_best.pth")

        # 7. 训练
        loss = agent.train_step()
        if loss is not None:
            recent_loss.append(loss)

        step_counter += N_ENVS
        total_steps += N_ENVS

        # 8. 打印日志（每2秒一次）
        now = time.time()
        if now - print_timer > 2.0:
            elapsed = now - start_time
            sps = step_counter / (now - print_timer)
            avg_score = sum(scores_window) / len(scores_window) if scores_window else 0
            max100 = max(scores_window) if scores_window else 0
            avg_loss = sum(recent_loss) / len(recent_loss) if recent_loss else 0
            cur_lr = agent.optimizer.param_groups[0]["lr"]

            # ★ 追踪最佳平均分
            if len(scores_window) >= 100 and avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save("snake_v6_rust_best_avg.pth")

            hours = elapsed / 3600
            remaining = (MAX_TRAINING_TIME - elapsed) / 3600

            print(
                f"Games: {total_games:>7} | Score: {avg_score:>6.2f} | Max100: {max100:>4} | "
                f"Rec: {record:>4} | Eps: {agent.epsilon:>6.4f} | "
                f"Loss: {avg_loss:>8.5f} | LR: {cur_lr:.7f} | "
                f"SPS: {sps:>7.0f} | Mem: {agent.memory.size:>8} | "
                f"Steps: {agent.steps_done:>7} | "
                f"Time: {hours:>5.2f}h (rem: {remaining:.2f}h)"
            )

            step_counter = 0
            print_timer = now

            # 定期保存（每2000局）
            if total_games > 0 and total_games % 2000 < N_ENVS * 2:
                agent.save("snake_v6_rust.pth")
                print(f"  >>> Checkpoint saved at game {total_games}")

        # ★ 每小时保存一次带时间戳的快照
        if total_games > 0 and elapsed > 0:
            hour_mark = int(elapsed // 3600)
            snapshot_file = f"snake_v6_rust_hour{hour_mark}.pth"
            # 简单判断：每小时第一次进入时保存
            if abs(elapsed - hour_mark * 3600) < 3.0 and not os.path.exists(snapshot_file):
                agent.save(snapshot_file)
                print(f"  >>> Hourly snapshot saved: {snapshot_file}")

    # 训练结束
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Total games: {total_games}, Total steps: {total_steps}")
    print(f"Training steps (steps_done): {agent.steps_done}")
    print(f"Best score: {record}")
    print(f"Best avg score (100-game window): {best_avg_score:.2f}")
    if scores_window:
        print(f"Last 100 avg: {sum(scores_window)/len(scores_window):.2f}, "
              f"Last 100 max: {max(scores_window)}")
    agent.save("snake_v6_rust_final.pth")
    print("Done! 🎉")


if __name__ == "__main__":
    train()