# Code was taken from Nicholas Renotte course for Reinforcement learning


import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        original = self.state
        self.state += action - 1
        self.shower_length -= 1

        reward = -1
        if 37 <= self.state <= 39:
            reward = 1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60
        return self.state


env = ShowerEnv()

episodes = 5
for episode in range(1 + episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
env.close()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
print(evaluate_policy(model, env, n_eval_episodes=100))

episodes = 1
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #print("Temperature: ", obs)
        #print("Action: ", action)
        #print("Reward: ", reward)
        #print("--------------")
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
env.close()

del model
