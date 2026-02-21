import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import random

# --- 导入 Rust 引擎 ---
try:
    import fast_snake_v6
except ImportError:
    print("错误: 找不到 fast_snake_v6 模块。")
    sys.exit(1)

# --- 全局配置 ---
MODEL_PATH = "snake_v6_rust_best.pth"
GRID_SIZE = 20
FPS_OPTIONS = [5, 8, 12, 18, 30]  # 5 档速度
FPS_LABELS = ["Slow", "Easy", "Normal", "Fast", "Insane"]

# 设计基准分辨率（所有布局基于此计算比例）
BASE_CELL = 25
BASE_BOARD = GRID_SIZE * BASE_CELL            # 500
BASE_PAD = 24                                 # 面板间距
BASE_SIDE_PAD = 28                            # 左右外边距
BASE_TOP_BAR = 110                            # 顶部栏
BASE_LABEL_H = 30                             # 标签区高度
BASE_BOTTOM_BAR = 56                          # 底部速度栏高度
BASE_W = BASE_SIDE_PAD * 2 + BASE_BOARD * 2 + BASE_PAD   # 1080
BASE_H = BASE_TOP_BAR + BASE_LABEL_H + BASE_BOARD + BASE_BOTTOM_BAR  # 696
BASE_ASPECT = BASE_W / BASE_H

# 最小窗口尺寸
MIN_WIN_W = 720
MIN_WIN_H = 460

# 颜色定义
C_BG          = (22, 22, 30)
C_BOARD_BG    = (32, 32, 40)
C_BOARD_DEAD  = (45, 35, 35)
C_GRID_LINE   = (44, 44, 55)
C_BORDER      = (58, 58, 70)
C_BORDER_DEAD = (120, 50, 50)

C_AI_HEAD     = (0, 230, 118)
C_AI_TAIL     = (200, 255, 220)
C_PL_HEAD     = (30, 144, 255)
C_PL_TAIL     = (200, 220, 255)

C_FOOD_MAIN   = (255, 70, 70)
C_FOOD_SHINE  = (255, 180, 180)
C_TEXT_MAIN   = (235, 235, 240)
C_TEXT_SUB    = (155, 155, 165)
C_TEXT_DIM    = (100, 100, 110)
C_TOPBAR_BG   = (28, 28, 36)
C_BOTTOMBAR_BG = (26, 26, 34)
C_DIVIDER     = (55, 55, 65)
C_BTN_NORMAL  = (44, 44, 56)
C_BTN_HOVER   = (64, 64, 78)
C_BTN_BORDER  = (100, 100, 115)
C_BTN_QUIT_N  = (70, 35, 35)
C_BTN_QUIT_H  = (110, 45, 45)
C_BTN_QUIT_B  = (160, 60, 60)
C_OVERLAY_BG  = (10, 10, 14)
C_SPEED_OFF   = (50, 50, 60)
C_SPEED_ON    = (80, 180, 255)
C_SPEED_DOT_OFF = (65, 65, 75)
C_SPEED_KNOB  = (255, 255, 255)

N_SPATIAL_CHANNELS = 3
N_AUX_FEATURES = 21
N_ACTIONS = 3


# ═══════════════════════════════════════════════
#  1. 网络架构 (保持不变)
# ═══════════════════════════════════════════════
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

    def forward(self, spatial, aux):
        x = self.pool(F.relu(self.conv1(spatial)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        a = F.relu(self.aux_fc1(aux))
        a = F.relu(self.aux_fc2(a))
        combined = torch.cat([x, a], dim=1)
        v = F.relu(self.val_ln(self.val_fc1(combined)))
        v = F.relu(self.val_fc2(v))
        v = self.val_out(v)
        adv = F.relu(self.adv_ln(self.adv_fc1(combined)))
        adv = F.relu(self.adv_fc2(adv))
        adv = self.adv_out(adv)
        return v + adv - adv.mean(dim=1, keepdim=True)


# ═══════════════════════════════════════════════
#  2. 玩家蛇逻辑
# ═══════════════════════════════════════════════
class PlayerSnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        self.food = self._place_food()
        self.score = 0
        self.is_dead = False

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def change_direction(self, new_dir):
        if (new_dir[0] + self.direction[0] != 0) or \
           (new_dir[1] + self.direction[1] != 0):
            self.next_direction = new_dir

    def step(self):
        if self.is_dead:
            return
        self.direction = self.next_direction
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)
        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            self.is_dead = True
            return
        if new_head in self.snake[:-1]:
            self.is_dead = True
            return
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

    def get_draw_data(self):
        body_with_val = []
        n = len(self.snake)
        for i, pos in enumerate(self.snake):
            val = 1.0 - (i / (n + 1)) * 0.5
            body_with_val.append((pos, val))
        return self.snake[0], body_with_val, self.food


# ═══════════════════════════════════════════════
#  3. 辅助渲染函数
# ═══════════════════════════════════════════════
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def draw_rounded_rect(surface, color, rect, radius):
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    if w <= 0 or h <= 0:
        return
    r = min(radius, w // 2, h // 2)
    if r < 1:
        pygame.draw.rect(surface, color, (x, y, w, h))
        return
    pygame.draw.rect(surface, color, (x + r, y, w - 2 * r, h))
    pygame.draw.rect(surface, color, (x, y + r, w, h - 2 * r))
    pygame.draw.circle(surface, color, (x + r, y + r), r)
    pygame.draw.circle(surface, color, (x + w - r, y + r), r)
    pygame.draw.circle(surface, color, (x + r, y + h - r), r)
    pygame.draw.circle(surface, color, (x + w - r, y + h - r), r)


def draw_board(surface, bx, by, board_px, cell_px, head, body_parts, food,
               theme_head, theme_tail, is_dead):
    bx, by, board_px = float(bx), float(by), float(board_px)
    cell_px = float(cell_px)

    bg = C_BOARD_DEAD if is_dead else C_BOARD_BG
    draw_rounded_rect(surface, bg, (bx, by, board_px, board_px), 8)

    # 网格线
    for i in range(1, GRID_SIZE):
        gx = int(bx + i * cell_px)
        gy = int(by + i * cell_px)
        pygame.draw.line(surface, C_GRID_LINE,
                         (gx, int(by) + 4), (gx, int(by + board_px) - 4))
        pygame.draw.line(surface, C_GRID_LINE,
                         (int(bx) + 4, gy), (int(bx + board_px) - 4, gy))

    border_c = C_BORDER_DEAD if is_dead else C_BORDER
    pygame.draw.rect(surface, border_c,
                     (int(bx), int(by), int(board_px), int(board_px)),
                     2, border_radius=8)

    # 食物
    if food:
        fx, fy = food
        cx = bx + fx * cell_px + cell_px / 2
        cy = by + fy * cell_px + cell_px / 2
        r = cell_px * 0.42
        pygame.draw.circle(surface, C_FOOD_MAIN, (int(cx), int(cy)), max(int(r), 2))
        pygame.draw.circle(surface, C_FOOD_SHINE,
                           (int(cx - r * 0.3), int(cy - r * 0.3)),
                           max(int(r * 0.28), 1))

    dead_color = (95, 95, 95)
    dead_head_color = (75, 75, 75)

    sorted_parts = sorted(body_parts, key=lambda p: p[1])
    total = len(sorted_parts)
    for idx, (pos, val) in enumerate(sorted_parts):
        px, py = pos
        if pos == head:
            continue
        ratio = idx / max(total - 1, 1)
        color = lerp_color(theme_tail, theme_head, ratio) if not is_dead else dead_color
        inset = max(int(cell_px * 0.06), 1)
        rect = (
            bx + px * cell_px + inset,
            by + py * cell_px + inset,
            cell_px - inset * 2,
            cell_px - inset * 2,
        )
        draw_rounded_rect(surface, color, rect, max(int(cell_px * 0.18), 2))

    if head:
        hx, hy = head
        color = theme_head if not is_dead else dead_head_color
        inset = max(int(cell_px * 0.04), 1)
        rect = (
            bx + hx * cell_px + inset,
            by + hy * cell_px + inset,
            cell_px - inset * 2,
            cell_px - inset * 2,
        )
        draw_rounded_rect(surface, color, rect, max(int(cell_px * 0.22), 2))

        eye_r = max(int(cell_px * 0.12), 2)
        ex1 = int(bx + hx * cell_px + cell_px * 0.32)
        ex2 = int(bx + hx * cell_px + cell_px * 0.68)
        ey = int(by + hy * cell_px + cell_px * 0.34)
        pygame.draw.circle(surface, (0, 0, 0), (ex1, ey), eye_r)
        pygame.draw.circle(surface, (0, 0, 0), (ex2, ey), eye_r)
        pr = max(eye_r // 2, 1)
        pygame.draw.circle(surface, (255, 255, 255), (ex1 + 1, ey - 1), pr)
        pygame.draw.circle(surface, (255, 255, 255), (ex2 + 1, ey - 1), pr)

    if is_dead:
        overlay = pygame.Surface((int(board_px), int(board_px)), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 60))
        surface.blit(overlay, (int(bx), int(by)))


# ═══════════════════════════════════════════════
#  4. 主游戏类
# ═══════════════════════════════════════════════
class BattleGame:
    def __init__(self):
        pygame.init()
        self.win_w = BASE_W
        self.win_h = BASE_H
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        pygame.display.set_caption("Snake AI vs Human - Battle Mode")
        self.clock = pygame.time.Clock()

        # 速度档位 (默认第3档 Normal)
        self.speed_index = 2
        self.current_fps = FPS_OPTIONS[self.speed_index]

        # 布局变量（由 _update_layout 填充）
        self.scale = 1.0
        self.offset_x = 0.0  # 居中偏移
        self._update_layout()

        # AI
        self.device = torch.device("cpu")
        self.ai_model = HybridDuelingQNet().to(self.device)
        self.load_ai_model()
        self.ai_env = fast_snake_v6.VectorizedSnakeEnv(1)

        # Player
        self.player_game = PlayerSnakeGame()

        # 状态
        self.state = "MENU"
        self.ai_wins = 0
        self.player_wins = 0
        self.countdown_val = 3
        self.countdown_start_tick = 0
        self.ai_dead = False
        self.ai_last_draw_data = None
        self.ai_score = 0

        # 速度条交互
        self._speed_dragging = False
        self._speed_bar_rects = []  # 每档的点击区域 (屏幕坐标)

    # ──────────── 布局计算 ────────────
    def _update_layout(self):
        """
        自由拉伸策略：
        - 根据窗口宽高，计算能放下的最大等比内容区域
        - 内容区域居中显示
        - 两侧/上下多余空间用背景色填充
        """
        w, h = self.win_w, self.win_h

        # 按宽高比计算可用的内容尺寸
        content_aspect = BASE_W / BASE_H
        if w / h > content_aspect:
            # 窗口比内容更宽 → 以高度为基准
            content_h = h
            content_w = h * content_aspect
        else:
            # 窗口比内容更高（或刚好）→ 以宽度为基准
            content_w = w
            content_h = w / content_aspect

        self.scale = content_w / BASE_W
        self.offset_x = (w - content_w) / 2
        self.offset_y = (h - content_h) / 2
        self.content_w = content_w
        self.content_h = content_h

        s = self.scale
        self.side_pad = BASE_SIDE_PAD * s
        self.mid_pad = BASE_PAD * s
        self.top_bar_h = BASE_TOP_BAR * s
        self.label_h = BASE_LABEL_H * s
        self.bottom_bar_h = BASE_BOTTOM_BAR * s

        self.board_px = (content_w - self.side_pad * 2 - self.mid_pad) / 2
        self.cell_px = self.board_px / GRID_SIZE

        self.board_y = self.offset_y + self.top_bar_h + self.label_h
        self.board1_x = self.offset_x + self.side_pad
        self.board2_x = self.offset_x + self.side_pad + self.board_px + self.mid_pad

        # 底部栏 y
        self.bottom_bar_y = self.board_y + self.board_px

        # 字体
        def sz(base):
            return max(int(base * s), 9)

        self.font_title = pygame.font.SysFont("Arial", sz(48), bold=True)
        self.font_score = pygame.font.SysFont("Arial", sz(36), bold=True)
        self.font_sub = pygame.font.SysFont("Arial", sz(20))
        self.font_label = pygame.font.SysFont("Arial", sz(17), bold=True)
        self.font_countdown = pygame.font.SysFont("Arial", sz(100), bold=True)
        self.font_overlay_title = pygame.font.SysFont("Arial", sz(52), bold=True)
        self.font_overlay_btn = pygame.font.SysFont("Arial", sz(26), bold=True)
        self.font_speed = pygame.font.SysFont("Arial", sz(14), bold=True)
        self.font_speed_label = pygame.font.SysFont("Arial", sz(16), bold=True)

        # 预计算速度条位置
        self._calc_speed_bar_rects()

    def _calc_speed_bar_rects(self):
        """计算速度选择器的5个档位点击区域。"""
        s = self.scale
        bar_cx = self.offset_x + self.content_w / 2
        bar_cy = self.bottom_bar_y + self.bottom_bar_h / 2
        total_w = 280 * s
        step = total_w / (len(FPS_OPTIONS) - 1)
        start_x = bar_cx - total_w / 2

        self._speed_bar_rects = []
        self._speed_dot_centers = []
        self._speed_track_start = (start_x, bar_cy)
        self._speed_track_end = (start_x + total_w, bar_cy)
        self._speed_track_w = total_w
        self._speed_step = step

        for i in range(len(FPS_OPTIONS)):
            cx = start_x + i * step
            hit_w = step * 0.8 if i < len(FPS_OPTIONS) - 1 else step * 0.4
            hit_rect = pygame.Rect(int(cx - hit_w / 2), int(bar_cy - 18 * s),
                                   int(hit_w), int(36 * s))
            self._speed_bar_rects.append(hit_rect)
            self._speed_dot_centers.append((cx, bar_cy))

    def _handle_resize(self, new_w, new_h):
        new_w = max(new_w, MIN_WIN_W)
        new_h = max(new_h, MIN_WIN_H)
        self.win_w = new_w
        self.win_h = new_h
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        self._update_layout()

    # ──────────── AI ────────────
    def load_ai_model(self):
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            self.ai_model.load_state_dict(ckpt["model"])
            self.ai_model.eval()
            print("AI Model Loaded.")
        else:
            print("Warning: Model not found.")

    def parse_ai_state(self, spatial_batch):
        spatial = spatial_batch[0]
        head_pos = np.where(spatial[0] > 0.5)
        head = (head_pos[1][0], head_pos[0][0]) if len(head_pos[0]) > 0 else None
        food_pos = np.where(spatial[2] > 0.5)
        food = (food_pos[1][0], food_pos[0][0]) if len(food_pos[0]) > 0 else None
        body_indices = np.where(spatial[1] > 0)
        body_parts = []
        for y, x in zip(body_indices[0], body_indices[1]):
            val = spatial[1][y][x]
            body_parts.append(((x, y), val))
        body_parts.sort(key=lambda p: p[1], reverse=True)
        return head, body_parts, food

    # ──────────── 游戏流程 ────────────
    def reset_round(self):
        self.ai_env.reset_all()
        self.player_game.reset()
        self.ai_dead = False
        self.ai_score = 0
        self.ai_last_draw_data = None
        self.state = "COUNTDOWN"
        self.countdown_val = 3
        self.countdown_start_tick = pygame.time.get_ticks()

    def set_speed(self, idx):
        idx = max(0, min(idx, len(FPS_OPTIONS) - 1))
        self.speed_index = idx
        self.current_fps = FPS_OPTIONS[idx]

    def update(self):
        if self.state == "COUNTDOWN":
            diff = pygame.time.get_ticks() - self.countdown_start_tick
            if diff < 1000:
                self.countdown_val = 3
            elif diff < 2000:
                self.countdown_val = 2
            elif diff < 3000:
                self.countdown_val = 1
            else:
                self.state = "PLAYING"

        if self.state == "PLAYING":
            if not self.ai_dead:
                spatial_np, aux_np = self.ai_env.get_states_batch()
                self.ai_last_draw_data = self.parse_ai_state(spatial_np)
                s_t = torch.from_numpy(spatial_np).to(self.device)
                a_t = torch.from_numpy(aux_np).to(self.device)
                with torch.no_grad():
                    q = self.ai_model(s_t, a_t)
                    action = q.argmax(dim=1).item()
                _, dones, scores = self.ai_env.step_batch([action])
                self.ai_score = int(scores[0])
                if dones[0]:
                    self.ai_dead = True

            if not self.player_game.is_dead:
                self.player_game.step()

            if self.ai_dead and self.player_game.is_dead:
                self.state = "GAMEOVER"
                if self.player_game.score > self.ai_score:
                    self.player_wins += 1
                elif self.ai_score > self.player_game.score:
                    self.ai_wins += 1

    # ──────────── 绘制 ────────────
    def draw_ui(self):
        self.screen.fill(C_BG)
        self._draw_top_bar()
        self._draw_labels()
        self._draw_boards()
        self._draw_bottom_bar()
        self._draw_state_overlay()
        pygame.display.flip()

    def _draw_top_bar(self):
        s = self.scale
        ox = self.offset_x
        oy = self.offset_y

        # 背景
        bar_rect = pygame.Rect(int(ox), int(oy),
                               int(self.content_w), int(self.top_bar_h))
        draw_rounded_rect(self.screen, C_TOPBAR_BG,
                          (ox, oy, self.content_w, self.top_bar_h), 0)
        pygame.draw.line(
            self.screen, C_DIVIDER,
            (int(ox), int(oy + self.top_bar_h) - 1),
            (int(ox + self.content_w), int(oy + self.top_bar_h) - 1),
        )

        cx = ox + self.content_w / 2

        # "VS"
        vs_surf = self.font_score.render("VS", True, C_TEXT_DIM)
        self.screen.blit(vs_surf, (int(cx - vs_surf.get_width() / 2),
                                   int(oy + self.top_bar_h * 0.18)))

        # AI 侧
        ai_cx = self.board1_x + self.board_px / 2
        ai_title = self.font_score.render("AI", True, C_AI_HEAD)
        self.screen.blit(ai_title, (int(ai_cx - ai_title.get_width() / 2),
                                    int(oy + self.top_bar_h * 0.10)))
        ai_w = self.font_sub.render(f"Wins: {self.ai_wins}", True, C_TEXT_SUB)
        self.screen.blit(ai_w, (int(ai_cx - ai_w.get_width() / 2),
                                int(oy + self.top_bar_h * 0.52)))
        ai_sc = self.font_sub.render(f"Score: {self.ai_score}", True, C_TEXT_MAIN)
        self.screen.blit(ai_sc, (int(ai_cx - ai_sc.get_width() / 2),
                                 int(oy + self.top_bar_h * 0.74)))

        # Player 侧
        pl_cx = self.board2_x + self.board_px / 2
        pl_title = self.font_score.render("PLAYER", True, C_PL_HEAD)
        self.screen.blit(pl_title, (int(pl_cx - pl_title.get_width() / 2),
                                    int(oy + self.top_bar_h * 0.10)))
        pl_w = self.font_sub.render(f"Wins: {self.player_wins}", True, C_TEXT_SUB)
        self.screen.blit(pl_w, (int(pl_cx - pl_w.get_width() / 2),
                                int(oy + self.top_bar_h * 0.52)))
        pl_sc = self.font_sub.render(
            f"Score: {self.player_game.score}", True, C_TEXT_MAIN)
        self.screen.blit(pl_sc, (int(pl_cx - pl_sc.get_width() / 2),
                                 int(oy + self.top_bar_h * 0.74)))

    def _draw_labels(self):
        y = int(self.offset_y + self.top_bar_h + self.label_h * 0.15)
        lbl_ai = self.font_label.render("AI  (Green)", True, C_AI_HEAD)
        self.screen.blit(lbl_ai, (int(self.board1_x + 6), y))
        lbl_pl = self.font_label.render(
            "Player  (Blue) — WASD / Arrows", True, C_PL_HEAD)
        self.screen.blit(lbl_pl, (int(self.board2_x + 6), y))

    def _draw_boards(self):
        bx1 = self.board1_x
        bx2 = self.board2_x
        by = self.board_y
        bp = self.board_px
        cp = self.cell_px

        if self.state == "MENU":
            draw_rounded_rect(self.screen, C_BOARD_BG,
                              (bx1, by, bp, bp), 8)
            pygame.draw.rect(self.screen, C_BORDER,
                             (int(bx1), int(by), int(bp), int(bp)),
                             2, border_radius=8)
            draw_rounded_rect(self.screen, C_BOARD_BG,
                              (bx2, by, bp, bp), 8)
            pygame.draw.rect(self.screen, C_BORDER,
                             (int(bx2), int(by), int(bp), int(bp)),
                             2, border_radius=8)
        else:
            if self.ai_last_draw_data:
                h, b, f = self.ai_last_draw_data
                draw_board(self.screen, bx1, by, bp, cp,
                           h, list(b), f, C_AI_HEAD, C_AI_TAIL, self.ai_dead)
            else:
                draw_rounded_rect(self.screen, C_BOARD_BG,
                                  (bx1, by, bp, bp), 8)
            ph, pb, pf = self.player_game.get_draw_data()
            draw_board(self.screen, bx2, by, bp, cp,
                       ph, pb, pf, C_PL_HEAD, C_PL_TAIL,
                       self.player_game.is_dead)

    def _draw_bottom_bar(self):
        """底部速度选择栏 —— 精致的5档滑块。"""
        s = self.scale
        ox = self.offset_x
        bar_y = self.bottom_bar_y
        bar_h = self.bottom_bar_h

        # 背景
        draw_rounded_rect(self.screen, C_BOTTOMBAR_BG,
                          (ox, bar_y, self.content_w, bar_h), 0)
        pygame.draw.line(self.screen, C_DIVIDER,
                         (int(ox), int(bar_y)),
                         (int(ox + self.content_w), int(bar_y)))

        if not self._speed_dot_centers:
            return

        # "SPEED" 标签
        speed_lbl = self.font_speed_label.render("SPEED", True, C_TEXT_DIM)
        lbl_x = self._speed_dot_centers[0][0] - speed_lbl.get_width() - 18 * s
        lbl_y = self._speed_dot_centers[0][1] - speed_lbl.get_height() / 2
        self.screen.blit(speed_lbl, (int(lbl_x), int(lbl_y)))

        # 轨道 (底部线)
        track_y = int(self._speed_dot_centers[0][1])
        track_x1 = int(self._speed_dot_centers[0][0])
        track_x2 = int(self._speed_dot_centers[-1][0])
        pygame.draw.line(self.screen, C_SPEED_OFF,
                         (track_x1, track_y), (track_x2, track_y),
                         max(int(3 * s), 2))

        # 激活段
        active_x = int(self._speed_dot_centers[self.speed_index][0])
        pygame.draw.line(self.screen, C_SPEED_ON,
                         (track_x1, track_y), (active_x, track_y),
                         max(int(3 * s), 2))

        # 各档位圆点 + 标签
        dot_r = max(int(5 * s), 3)
        knob_r = max(int(8 * s), 5)
        mouse_pos = pygame.mouse.get_pos()

        for i, (dcx, dcy) in enumerate(self._speed_dot_centers):
            dcx_i, dcy_i = int(dcx), int(dcy)

            if i == self.speed_index:
                # 当前选中 —— 大亮色圆点
                pygame.draw.circle(self.screen, C_SPEED_ON,
                                   (dcx_i, dcy_i), knob_r)
                pygame.draw.circle(self.screen, C_SPEED_KNOB,
                                   (dcx_i, dcy_i), max(int(4 * s), 2))
            elif i < self.speed_index:
                # 已过的档位
                pygame.draw.circle(self.screen, C_SPEED_ON,
                                   (dcx_i, dcy_i), dot_r)
            else:
                # 未到的档位
                pygame.draw.circle(self.screen, C_SPEED_DOT_OFF,
                                   (dcx_i, dcy_i), dot_r)

            # 档位标签
            lbl = self.font_speed.render(FPS_LABELS[i], True,
                                         C_TEXT_MAIN if i == self.speed_index
                                         else C_TEXT_DIM)
            self.screen.blit(lbl, (int(dcx - lbl.get_width() / 2),
                                   int(dcy + 12 * s)))

        # FPS 数字显示在右侧
        fps_txt = self.font_speed_label.render(
            f"{self.current_fps} FPS", True, C_SPEED_ON)
        fps_x = self._speed_dot_centers[-1][0] + 22 * s
        fps_y = self._speed_dot_centers[-1][1] - fps_txt.get_height() / 2
        self.screen.blit(fps_txt, (int(fps_x), int(fps_y)))

    def _draw_state_overlay(self):
        if self.state == "MENU":
            self._draw_overlay_menu()
        elif self.state == "COUNTDOWN":
            self._draw_countdown()
        elif self.state == "GAMEOVER":
            self._draw_overlay_gameover()

    def _draw_overlay_menu(self):
        self._draw_overlay_bg()
        cx = self.offset_x + self.content_w / 2
        cy = self.offset_y + self.content_h / 2
        s = self.scale

        t_surf = self.font_overlay_title.render("SNAKE BATTLE", True, C_TEXT_MAIN)
        self.screen.blit(t_surf, (int(cx - t_surf.get_width() / 2),
                                  int(cy - 70 * s)))

        self._draw_button("Press SPACE to Start", cx, cy + 16 * s,
                          C_BTN_NORMAL, C_BTN_HOVER, C_BTN_BORDER)

    def _draw_overlay_gameover(self):
        self._draw_overlay_bg()
        cx = self.offset_x + self.content_w / 2
        cy = self.offset_y + self.content_h / 2
        s = self.scale

        if self.player_game.score > self.ai_score:
            title, color = "PLAYER WINS!", C_PL_HEAD
        elif self.ai_score > self.player_game.score:
            title, color = "AI WINS!", C_AI_HEAD
        else:
            title, color = "DRAW", C_TEXT_MAIN

        t_surf = self.font_overlay_title.render(title, True, color)
        self.screen.blit(t_surf, (int(cx - t_surf.get_width() / 2),
                                  int(cy - 90 * s)))

        # Restart 按钮
        btn_y = cy - 6 * s
        self._restart_btn_rect = self._draw_button(
            "Press SPACE to Restart", cx, btn_y,
            C_BTN_NORMAL, C_BTN_HOVER, C_BTN_BORDER)

        # Quit 按钮
        quit_y = btn_y + 66 * s
        self._quit_btn_rect = self._draw_button(
            "Quit Game (ESC)", cx, quit_y,
            C_BTN_QUIT_N, C_BTN_QUIT_H, C_BTN_QUIT_B)

    def _draw_overlay_bg(self):
        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        overlay.fill((C_OVERLAY_BG[0], C_OVERLAY_BG[1], C_OVERLAY_BG[2], 190))
        self.screen.blit(overlay, (0, 0))

    def _draw_button(self, text, cx, cy, color_n, color_h, color_b):
        """绘制居中按钮，返回 Rect。"""
        s = self.scale
        btn_w = int(380 * s)
        btn_h = int(52 * s)
        btn_rect = pygame.Rect(int(cx - btn_w / 2), int(cy), btn_w, btn_h)
        mouse_pos = pygame.mouse.get_pos()
        btn_color = color_h if btn_rect.collidepoint(mouse_pos) else color_n
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, color_b, btn_rect, 2, border_radius=10)

        s_surf = self.font_overlay_btn.render(text, True, (225, 225, 230))
        self.screen.blit(s_surf, (int(cx - s_surf.get_width() / 2),
                                  int(btn_rect.y + btn_h / 2 -
                                      s_surf.get_height() / 2)))
        return btn_rect

    def _draw_countdown(self):
        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        txt = str(self.countdown_val)
        colors = {3: (255, 220, 60), 2: (255, 160, 30), 1: (255, 60, 60)}
        color = colors.get(self.countdown_val, (255, 255, 255))

        t_surf = self.font_countdown.render(txt, True, color)
        cx = int(self.offset_x + self.content_w / 2)
        cy = int(self.board_y + self.board_px / 2)
        self.screen.blit(t_surf, (cx - t_surf.get_width() // 2,
                                  cy - t_surf.get_height() // 2))

    # ──────────── 速度条点击检测 ────────────
    def _handle_speed_click(self, pos):
        for i, rect in enumerate(self._speed_bar_rects):
            if rect.collidepoint(pos):
                self.set_speed(i)
                return True
        return False

    # ──���───────── 主循环 ────────────
    def run(self):
        running = True
        self._restart_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._quit_btn_rect = pygame.Rect(0, 0, 0, 0)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 速度条点击
                    self._handle_speed_click(event.pos)

                    # Game-over 按钮点击
                    if self.state == "GAMEOVER":
                        if self._restart_btn_rect.collidepoint(event.pos):
                            self.reset_round()
                        elif self._quit_btn_rect.collidepoint(event.pos):
                            running = False

                    # Menu 按钮点击 → start
                    if self.state == "MENU":
                        # 检测是否点在 start 按钮范围内
                        # （按钮区域在 draw 时才确定，这里简单地用 SPACE 逻辑的
                        #   补充：允许鼠标点击启动）
                        s = self.scale
                        cx = self.offset_x + self.content_w / 2
                        cy = self.offset_y + self.content_h / 2 + 16 * s
                        btn_w, btn_h = 380 * s, 52 * s
                        start_rect = pygame.Rect(
                            int(cx - btn_w / 2), int(cy),
                            int(btn_w), int(btn_h))
                        if start_rect.collidepoint(event.pos):
                            self.reset_round()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == "GAMEOVER":
                            running = False
                        elif self.state == "PLAYING":
                            # ESC during play → 回到菜单（可选行为）
                            self.state = "MENU"
                        else:
                            running = False

                    # SPACE → start / restart
                    if self.state in ("MENU", "GAMEOVER"):
                        if event.key == pygame.K_SPACE:
                            self.reset_round()

                    # 方向键 & WASD
                    if self.state == "PLAYING" and \
                       not self.player_game.is_dead:
                        if event.key in (pygame.K_w, pygame.K_UP):
                            self.player_game.change_direction((0, -1))
                        elif event.key in (pygame.K_s, pygame.K_DOWN):
                            self.player_game.change_direction((0, 1))
                        elif event.key in (pygame.K_a, pygame.K_LEFT):
                            self.player_game.change_direction((-1, 0))
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            self.player_game.change_direction((1, 0))

                    # 速度快捷键：1-5
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        self.set_speed(0)
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        self.set_speed(1)
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        self.set_speed(2)
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        self.set_speed(3)
                    elif event.key in (pygame.K_5, pygame.K_KP5):
                        self.set_speed(4)

            self.update()
            self.draw_ui()
            self.clock.tick(self.current_fps)

        pygame.quit()


if __name__ == "__main__":
    game = BattleGame()
    game.run()