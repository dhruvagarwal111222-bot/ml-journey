import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
import datetime

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# load weights
model = DQN()
weights_path = "Cartpole/cartpole_best.pth"
mod_time = os.path.getmtime(weights_path)
print(f"loading weights from: {weights_path}")
print(f"last saved: {datetime.datetime.fromtimestamp(mod_time)}")
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()

# run episodes
NUM_EPISODES = 10
env = gym.make("CartPole-v1")
all_episodes = []

print(f"\nrunning {NUM_EPISODES} episodes...\n")
for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    states = [state]
    done = False
    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_t).argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
    all_episodes.append(states)
    print(f"episode {ep+1:2d}: {len(states):4d} steps")

env.close()

avg = np.mean([len(ep) for ep in all_episodes])
best = max([len(ep) for ep in all_episodes])
worst = min([len(ep) for ep in all_episodes])
print(f"\navg: {avg:.0f}  |  best: {best}  |  worst: {worst}")

# flatten
flat_states = []
flat_ep_nums = []
flat_step_nums = []
for ep_idx, ep_states in enumerate(all_episodes):
    for step_idx, s in enumerate(ep_states):
        flat_states.append(s)
        flat_ep_nums.append(ep_idx + 1)
        flat_step_nums.append(step_idx)

# animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-0.3, 2.8)
ax.set_facecolor("#f0f4f8")
ax.axhline(y=0.1, color='#555', linewidth=3, zorder=1)
ax.axis('off')

cart_w, cart_h = 0.5, 0.25
cart = patches.FancyBboxPatch((0, 0.1), cart_w, cart_h,
    boxstyle="round,pad=0.03", color="#378ADD", zorder=3)
ax.add_patch(cart)

wheel_l = plt.Circle((0, 0.1), 0.07, color="#1a1a2e", zorder=4)
wheel_r = plt.Circle((0, 0.1), 0.07, color="#1a1a2e", zorder=4)
ax.add_patch(wheel_l)
ax.add_patch(wheel_r)

pole, = ax.plot([], [], color="#E24B4A", linewidth=7,
    solid_capstyle='round', zorder=5)
ball, = ax.plot([], [], 'o', color="#c0392b", markersize=12, zorder=6)

# progress bar background
progress_bg = patches.Rectangle((-2.8, -0.15), 5.6, 0.08,
    color='#ddd', zorder=2)
progress_bar = patches.Rectangle((-2.8, -0.15), 0, 0.08,
    color='#378ADD', zorder=3)
ax.add_patch(progress_bg)
ax.add_patch(progress_bar)

ep_text   = ax.text(0, 2.6, '', fontsize=14, color='#222',
    fontweight='bold', ha='center')
info_text = ax.text(0, 2.4, '', fontsize=10, color='#666', ha='center')
score_text = ax.text(0, 2.2, '', fontsize=10, color='#378ADD', ha='center')

pole_len = 1.0

def update(frame):
    s = flat_states[frame]
    cart_pos, _, pole_angle, _ = s
    ep_num = flat_ep_nums[frame]
    step_num = flat_step_nums[frame]
    ep_len = len(all_episodes[ep_num - 1])

    # cart and wheels
    cx = cart_pos - cart_w / 2
    cart.set_x(cx)
    cart.set_y(0.1)
    wheel_l.center = (cx + 0.1, 0.1)
    wheel_r.center = (cx + cart_w - 0.1, 0.1)

    # pole
    pivot_x = cart_pos
    pivot_y = 0.1 + cart_h
    tip_x = pivot_x + pole_len * np.sin(pole_angle)
    tip_y = pivot_y + pole_len * np.cos(pole_angle)
    pole.set_data([pivot_x, tip_x], [pivot_y, tip_y])
    ball.set_data([tip_x], [tip_y])

    # progress bar
    progress = step_num / ep_len
    progress_bar.set_width(5.6 * progress)

    # text
    ep_text.set_text(f"episode {ep_num} / {NUM_EPISODES}")
    info_text.set_text(f"step: {step_num} / {ep_len}  |  angle: {np.degrees(pole_angle):.1f}°  |  cart: {cart_pos:.2f}")
    score_text.set_text(f"this episode: {ep_len} steps  |  avg all episodes: {avg:.0f}")

    return cart, pole, ball, wheel_l, wheel_r, ep_text, info_text, score_text, progress_bar

ani = FuncAnimation(fig, update, frames=len(flat_states),
    interval=30, blit=True, repeat=False)

plt.tight_layout()
plt.show()