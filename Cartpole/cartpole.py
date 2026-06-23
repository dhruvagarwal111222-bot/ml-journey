import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

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

class Agent:
    def __init__(self):
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 20

        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return self.model(state_t).argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states))
        next_states_t = torch.FloatTensor(np.array(next_states))
        rewards_t     = torch.FloatTensor(rewards)
        dones_t       = torch.FloatTensor(dones)
        actions_t     = torch.LongTensor(actions)

        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # double DQN — online picks action, target evaluates it
        with torch.no_grad():
            next_actions = self.model(next_states_t).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states_t).gather(1, next_actions).squeeze()
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# training
env = gym.make("CartPole-v1")
agent = Agent()
scores = []
best_avg = 0

print("Double DQN training started...\n")

for episode in range(700):
    state, _ = env.reset()
    score = 0

    for _ in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        cart_pos = next_state[0]
        pole_angle = next_state[2]
        shaped_reward = (1.0 - 0.5 * abs(cart_pos) - 0.5 * abs(pole_angle)) if not done else -10

        agent.remember(state, action, shaped_reward, next_state, done)
        # in training loop replace agent.replay() with:
        if len(agent.memory) % 4 == 0:
            agent.replay()
        state = next_state
        score += 1
        if done:
            break

    scores.append(score)
    avg = np.mean(scores[-20:])

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % agent.update_target_every == 0:
        agent.update_target()

    if avg > best_avg:
        best_avg = avg
        torch.save(agent.model.state_dict(), "Cartpole/cartpole_best.pth")

    torch.save(agent.model.state_dict(), "Cartpole/cartpole_last.pth")

    print(f"Episode {episode+1:3d} | Score: {score:4d} | Avg(20): {avg:6.1f} | Epsilon: {agent.epsilon:.3f} | Best avg: {best_avg:.1f}")

    if avg >= 495:
        print(f"\n*** SOLVED at episode {episode+1}! ***")
        break

env.close()

plt.figure(figsize=(10, 5))
plt.plot(scores, alpha=0.4, label="score")
plt.plot(np.convolve(scores, np.ones(20)/20, mode='valid'), label="avg (20 ep)")
plt.axhline(y=495, color='r', linestyle='--', label="solved threshold")
plt.xlabel("episode")
plt.ylabel("score")
plt.title("CartPole — Double DQN")
plt.legend()
plt.tight_layout()
plt.savefig("Cartpole/cartpole_results.png")
plt.show()
print("\ndone! weights saved to Cartpole/cartpole_best.pth and Cartpole/cartpole_last.pth")