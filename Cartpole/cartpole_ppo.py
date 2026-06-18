import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"\navg reward: {mean_reward:.1f} +/- {std_reward:.1f}")

model.save("Cartpole/cartpole_ppo")
print("saved!")