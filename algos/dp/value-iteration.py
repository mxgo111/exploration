from os import system, name
import time
from time import sleep
from IPython.display import clear_output
import gym
import random
import numpy as np

import sys
sys.path.append('/Users/mxgo/rl/code/exploration/envs')

from frozen_lake import FrozenLakeEnv
from taxi import TaxiEnv

class ValueIteration:
    """
    Algorithm for Value Iteration
    """

    # Initialize policy, environment, value table (V)
    def __init__(self, env, policy=None, gamma=1.0):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = policy if policy else self.create_random_policy()
        self.V = np.zeros(self.num_states)
        self.gamma = gamma

    # Creates random policy
    def create_random_policy(self):
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    # finds the best action given value function
    def update_action_values(self, state):
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for prob, next_state, reward, end in self.env.P[state][action]:
                action_values[action] += prob * (reward + self.gamma * self.V[next_state])
        return action_values

    # Value iteration
    def value_iteration(self, theta=1e-9, terms=1e9):
        for i in range(int(terms)):
            delta = 0
            for state in range(self.num_states):
                action_values = self.update_action_values(state)
                best_action_value = np.max(action_values)
                delta = max(delta, np.abs(self.V[state] - best_action_value))
                self.V[state] = best_action_value
            # convergence of values
            if delta < theta:
                break

        # find deterministic best policy
        for state in range(self.num_states):
            action_values = self.update_action_values(state)
            best_action = np.argmax(action_values)
            self.policy[state] = best_action

    def play_game(self, display=True):
        self.env.reset()
        episodes = []
        finished = False

        curr_state = self.env.s
        total_reward = 0

        while not finished:
            # display current state
            if display:
                system('clear')
                clear_output(True)
                self.env.render()
                sleep(0.5)

            # find next state
            # s = self.env.env.s
            action = self.policy[curr_state]
            curr_state, reward, finished, info = self.env.step(action)
            total_reward += reward
            episodes.append([curr_state, action, reward])

        # display end result
        if display:
            system('clear')
            clear_output(True)
            self.env.render()

        print(f"Total Reward from this run: {total_reward}")
        return episodes

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                curr_state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        return np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)

    # prints reward information
    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        mean, var, best = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")

if __name__ == "__main__":
    # env = gym.make('FrozenLake-v0', is_slippery=False)
    # env = FrozenLakeEnv(map_name="2x2", is_slippery=False)
    # env = FrozenLakeEnv(map_name="16x16", is_slippery=True)
    # env = gym.make('Taxi-v3')
    env = TaxiEnv()
    print("num states", env.nS)
    env.reset()

    my_policy = ValueIteration(env, gamma=0.9)
    my_policy.value_iteration()
    my_policy.play_game()
    my_policy.print_rewards_info(num_episodes=1000)
