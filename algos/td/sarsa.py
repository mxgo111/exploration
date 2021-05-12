from os import system, name
import time
from time import sleep
from IPython.display import clear_output
import matplotlib.pyplot as plt
import gym
import random
import numpy as np

import sys
sys.path.append('/Users/mxgo/rl/code/exploration/envs')

from frozen_lake import FrozenLakeEnv

class SARSA:
    """
    Algorithm for the On-policy TD method SARSA
    """

    # Initialize policy, environment, value table (V)
    def __init__(self, env, gamma=1.0, alpha=0.1, epsilon=0.01):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        # Without model, states alone not sufficient -> requires action-values
        self.Q = np.zeros((self.num_states, self.num_actions))

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    # choose action from Q-values based on epsilon-greedy
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(np.arange(self.num_actions))
        else:
            best_actions = np.argwhere(self.Q[state] == np.amax(self.Q[state])).flatten()
            return np.random.choice(best_actions)

    # perform sarsa with epsilon greedy policy
    def sarsa(self, num_episodes=10000, interval=1000, display=False, step_limit=10000):
        mean_returns = []
        for e in range(1,num_episodes+1):
            self.epsilon = min(0.01, 100/(e+1))
            self.alpha = min(0.1, 1000/(e+1))
            self.env.reset()
            finished = False

            curr_state = self.env.s
            num_steps = 0

            action = self.choose_action(curr_state)
            while not finished and num_steps < step_limit:
                # display current state
                if display:
                    system('clear')
                    clear_output(True)
                    self.env.render()
                    sleep(1)

                # take a step
                next_state, reward, finished, info = self.env.step(action)

                # choose next action based on epsilon greedy policy
                next_action = self.choose_action(next_state)

                # update Q valuess
                self.Q[curr_state][action] = self.Q[curr_state][action] + \
                                             self.alpha * (reward + \
                                                           self.gamma * self.Q[next_state][next_action] - \
                                                           self.Q[curr_state][action])

                num_steps += 1
                curr_state = next_state
                action = next_action

            # run tests every interval episodes
            if e % interval == 0:
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)

        plt.plot(np.arange(interval, num_episodes+1, interval), mean_returns)
        plt.savefig("sarsa")
        plt.show()

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.choose_action(curr_state)
                curr_state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        mean, var, best = np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")
        return mean, var, best

if __name__ == "__main__":
    # env = gym.make('FrozenLake-v0', is_slippery=False)
    # env = FrozenLakeEnv(map_name="2x2", is_slippery=False)
    env = FrozenLakeEnv(map_name="8x8", is_slippery=True)
    # env = FrozenLakeEnv(map_name="16x16", is_slippery=True)
    # env = gym.make('Taxi-v3')
    print("num states", env.nS)
    env.reset()

    my_policy = SARSA(env, gamma=0.9, alpha=0.1, epsilon=0.01)
    my_policy.sarsa(num_episodes=50000)
