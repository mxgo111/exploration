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

class MonteCarloOnPolicyEspSoft:
    """
    Algorithm for First-Visit Monte Carlo with Epsilon Soft
    """

    # Initialize policy, environment, value table (V)
    def __init__(self, env, policy=None, gamma=1.0):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = policy if policy else self.create_random_policy()

        # Without model, states alone not sufficient -> requires action-values
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.N = np.zeros((self.num_states, self.num_actions))

        self.gamma = gamma

    # Creates random policy
    def create_random_policy(self):
        # policy is (num_states * num_actions) array (probabilistic)
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        return policy

    # perform monte carlo with epsilon soft policy
    def monte_carlo(self, epsilon=0.01, num_episodes=10000, interval=1000):
        mean_returns = []
        for e in range(1,num_episodes+1):
            epsilon = min(0.01, 100/(e+1))
            G = 0
            episode = self.play_game(display=False)
            episode_state_actions = [(x[0], x[1]) for x in episode]

            # run tests every interval episodes
            if e % interval == 0:
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)

            for t in reversed(range(len(episode))):
                s_t, a_t, r_t = episode[t]
                state_action = (s_t, a_t)
                G = self.gamma * G + r_t

                # using first visits
                if not state_action in episode_state_actions[:t]:
                    self.N[s_t][a_t] += 1
                    self.Q[s_t][a_t] += (1/self.N[s_t][a_t]) * (G - self.Q[s_t][a_t])

                    best_actions = np.argwhere(self.Q[s_t] == np.amax(self.Q[s_t])).flatten()
                    best_action = np.random.choice(best_actions)

                    self.policy[s_t] = np.ones(self.num_actions) * epsilon/self.num_actions
                    self.policy[s_t][best_action] = 1 - epsilon + epsilon/self.num_actions

        plt.plot(np.arange(interval, num_episodes+1, interval), mean_returns)
        plt.savefig("on-policy-eps-soft")
        plt.show()

    # run an episode and record state, action, reward info
    def play_game(self, display=True, step_limit=10000):
        self.env.reset()
        episode = []
        finished = False

        curr_state = self.env.s
        total_reward = 0
        num_steps = 0

        while not finished and num_steps < step_limit:
            # display current state
            if display:
                system('clear')
                clear_output(True)
                self.env.render()
                sleep(1)

            n = random.uniform(0, sum(self.policy[curr_state]))
            top_range = 0
            action = 0
            for i, prob in enumerate(self.policy[curr_state]):
                top_range += prob
                if n < top_range:
                    action = i
                    break

            next_state, reward, finished, info = self.env.step(action)
            total_reward += reward
            episode.append([curr_state, action, reward])
            num_steps += 1
            curr_state = next_state

        # display end result
        if display:
            system('clear')
            clear_output(True)
            self.env.render()
            print(f"Total Reward from this run: {total_reward}")

        return episode

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = np.argmax(self.policy[curr_state])
                curr_state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        mean, var, best = np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")
        return mean, var, best

if __name__ == "__main__":
    # env = gym.make('FrozenLake-v0', is_slippery=False)
    # env = FrozenLakeEnv(map_name="2x2", is_slippery=False)
    env = FrozenLakeEnv(map_name="8x8", is_slippery=False)
    # env = FrozenLakeEnv(map_name="16x16", is_slippery=True)
    # env = gym.make('Taxi-v3')
    print("num states", env.nS)
    env.reset()

    my_policy = MonteCarloOnPolicyEspSoft(env, gamma=0.9)
    my_policy.monte_carlo(num_episodes=50000)
