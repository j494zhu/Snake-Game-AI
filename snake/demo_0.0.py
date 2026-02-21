import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

# --- 导入你的 Rust 引擎 ---
# 必须确保编译好的 .so (Linux/Mac) 或 .pyd (Windows) 文件在当前目录
try:
    import fast_snake_v6
except ImportError:
    print("错误: 找不到 fast_snake_v6 模块。请确保编译好的 Rust 库在当前目录下。")
    sys.exit(1)

# --- 配置参数 ---
MODEL_PATH = "snake_v6_rust_best.pth"  # 优先加载最高分模型
# MODEL_PATH = "snake_v6_rust_best_avg.pth" # 也可以尝试这个，通常表现更稳定

GRID_SIZE = 20
CELL_SIZE = 30  # 每个格子的像素大小
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 15        # 蛇的移动速度，调大变快

# 这里的参数必须与训练时完全一致
N_SPATIAL_CHANNELS = 3
N_AUX_FEATURES = 21
N_ACTIONS = 3

# --- 1. 网络架构 (必须与训练代码完全一致) ---
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
        # Inference 时不需要 dropout，但为了保持一致性如果不调用 .eval() 会有影响
        # 这里我们会调用 model.eval()，所以 dropout 会自动关闭

        v = F.relu(self.val_ln(self.val_fc1(combined)))
        v = F.relu(self.val_fc2(v))
        v = self.val_out(v)

        adv = F.relu(self.adv_ln(self.adv_fc1(combined)))
        adv = F.relu(self.adv_fc2(adv))
        adv = self.adv_out(adv)

        return v + adv - adv.mean(dim=1, keepdim=True)

# --- 2. 改进后的解析函数 ---
def parse_state_with_values(spatial_batch):
    """
    返回:
    head: (x, y)
    body_parts: List of ((x, y), value) <- 注意这里带上了像素值用于排序
    food: (x, y)
    """
    spatial = spatial_batch[0] # 取出第一个环境
    
    # 1. 解析蛇头
    head_pos = np.where(spatial[0] > 0.5)
    head = (head_pos[1][0], head_pos[0][0]) if len(head_pos[0]) > 0 else None

    # 2. 解析食物
    food_pos = np.where(spatial[2] > 0.5)
    food = (food_pos[1][0], food_pos[0][0]) if len(food_pos[0]) > 0 else None

    # 3. 解析蛇身 (带数值)
    # Rust代码中: spatial[1] 的值是从 1.0 (颈部) 递减到 0.5 (尾部)
    # 我们利用这个性质来对蛇身进行排序
    body_indices = np.where(spatial[1] > 0)
    body_parts = []
    for y, x in zip(body_indices[0], body_indices[1]):
        val = spatial[1][y][x]
        body_parts.append(((x, y), val))
    
    # ★ 关键步骤：按数值从大到小排序 (1.0是脖子，0.5是尾巴)
    body_parts.sort(key=lambda x: x[1], reverse=True)
    
    return head, body_parts, food

# --- 辅助函数：颜色插值 ---
def get_gradient_color(start_col, end_col, ratio):
    """ ratio: 0.0 -> start_col, 1.0 -> end_col """
    r = start_col[0] + (end_col[0] - start_col[0]) * ratio
    g = start_col[1] + (end_col[1] - start_col[1]) * ratio
    b = start_col[2] + (end_col[2] - start_col[2]) * ratio
    return (int(r), int(g), int(b))

# --- 3. 主程序 (带渐变色) ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))
    pygame.display.set_caption("Snake AI - Gradient Color Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    # === 颜色配置 ===
    COLOR_BG = (20, 20, 20)           # 背景更黑一点，突出渐变
    COLOR_GRID = (35, 35, 35)
    
    COLOR_HEAD = (0, 255, 0)          # 纯绿蛇头
    COLOR_BODY_START = (0, 200, 0)    # 蛇身开始的颜色 (深绿)
    COLOR_BODY_END = (255, 255, 255)  # 蛇身结束的颜色 (纯白)
    
    COLOR_FOOD = (255, 60, 60)
    COLOR_TEXT = (200, 200, 200)

    device = torch.device("cpu")
    model = HybridDuelingQNet().to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Model not found!")
        return

    model.eval()
    env = fast_snake_v6.VectorizedSnakeEnv(1)
    env.reset_all()

    game_count = 1
    current_score = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        # 获取状态
        spatial_np, aux_np = env.get_states_batch()
        spatial_t = torch.from_numpy(spatial_np).to(device)
        aux_t = torch.from_numpy(aux_np).to(device)

        # AI 决策
        with torch.no_grad():
            q_vals = model(spatial_t, aux_t)
            action = q_vals.argmax(dim=1).item()

        # 环境步进
        _, dones, scores = env.step_batch([action])
        
        if dones[0]:
            print(f"Game {game_count} Finished! Score: {int(scores[0])}")
            game_count += 1
            current_score = 0
        else:
            current_score = int(scores[0])

        # 解析 (使用新的带排序功能的函数)
        head, sorted_body, food = parse_state_with_values(spatial_np)

        # === 绘图 ===
        screen.fill(COLOR_BG)

        # 画网格
        for i in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(screen, COLOR_GRID, (i, 0), (i, WINDOW_SIZE))
            pygame.draw.line(screen, COLOR_GRID, (0, i), (WINDOW_SIZE, i))

        # 1. 画蛇身 (渐变)
        total_segments = len(sorted_body)
        for i, ((bx, by), val) in enumerate(sorted_body):
            # 计算渐变比例 (0.0 是脖子, 1.0 是尾巴尖)
            ratio = i / max(total_segments, 1)
            
            # 获取当前节的颜色
            segment_color = get_gradient_color(COLOR_BODY_START, COLOR_BODY_END, ratio)
            
            rect = (bx * CELL_SIZE, by * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # 画实心矩形
            pygame.draw.rect(screen, segment_color, rect)
            # 画一个极细的深色描边，让身体关节分明一点
            pygame.draw.rect(screen, (20, 20, 20), rect, 1)

        # 2. 画蛇头 (覆盖在第一节身体上)
        if head:
            hx, hy = head
            # 蛇头画稍微大一点点，或者加个特效
            rect = (hx * CELL_SIZE, hy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_HEAD, rect)
            
            # 简单的眼睛
            eye_color = (0, 0, 0)
            pygame.draw.circle(screen, eye_color, (hx*CELL_SIZE+8, hy*CELL_SIZE+8), 4)
            pygame.draw.circle(screen, eye_color, (hx*CELL_SIZE+22, hy*CELL_SIZE+8), 4)

        # 3. 画食物 (带一点呼吸灯效果的圆)
        if food:
            fx, fy = food
            center = (fx * CELL_SIZE + CELL_SIZE//2, fy * CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(screen, COLOR_FOOD, center, CELL_SIZE//2 - 2)
            pygame.draw.circle(screen, (255, 150, 150), center, CELL_SIZE//4)

        # UI
        pygame.draw.rect(screen, (40, 40, 40), (0, WINDOW_SIZE, WINDOW_SIZE, 40))
        text = f"Game: {game_count} | Score: {current_score} | Best: {checkpoint.get('record', '?')}"
        screen.blit(font.render(text, True, COLOR_TEXT), (15, WINDOW_SIZE + 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()