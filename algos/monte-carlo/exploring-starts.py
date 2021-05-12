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

class MonteCarloExploringStarts:
    """
    Algorithm for Monte Carlo with Exploring Starts
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
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    # perform monte carlo with exploring starts
    def monte_carlo_exploring_starts(self, num_episodes=100):

        returns = {}
        for e in range(1,num_episodes+1):
            G = 0
            episode = self.play_game(display=False)
            episode_state_actions = [(x[0], x[1]) for x in episode]

            if e % 1000 == 0:
                self.print_rewards_info(num_episodes=100)
                print(self.Q)
                print(self.policy)
                # import sys; sys.exit()

            for t in reversed(range(len(episode))):
                s_t, a_t, r_t = episode[t]
                # if s_t == 14 and a_t == 2:
                #     print("HAHAH")
                # if r_t > 0:
                #     print(s_t, a_t, r_t)
                state_action = (s_t, a_t)
                G = self.gamma * G + r_t

                # using first visits
                if not state_action in episode_state_actions[:t]:
                    if returns.get(state_action):
                        returns[state_action].append(G)
                    else:
                        returns[state_action] = [G]

                    self.Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])

                    best_actions = np.argwhere(self.Q[s_t] == np.amax(self.Q[s_t])).flatten()
                    self.policy[s_t] = np.random.choice(best_actions)

                    # self.N[s_t,a_t] += 1
                    # self.Q[s_t,a_t] += (G - self.Q[s_t,a_t])/self.N[s_t,a_t]

                    # best_actions = np.argwhere(self.Q[s_t] == np.amax(self.Q[s_t])).flatten()
                    # self.policy[s_t] = np.random.choice(best_actions)

    def play_game(self, display=True, step_limit=100):
        self.env.reset()
        episode = []
        finished = False

        curr_state = self.env.s
        # print(self.env.s)
        total_reward = 0
        num_steps = 0

        while not finished and num_steps < step_limit:
            # display current state
            if display:
                system('clear')
                clear_output(True)
                self.env.render()
                sleep(1)

            # find next state
            if not episode:
                action = np.random.choice(range(self.num_actions))
                # print("HELLO")
            else:
                action = self.policy[curr_state]
            curr_state, reward, finished, info = self.env.step(action)
            total_reward += reward
            episode.append([curr_state, action, reward])
            # print("in progress")
            num_steps += 1

        # print("finished")

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
    # env = FrozenLakeEnv(map_name="4x4", is_slippery=False)
    # env = FrozenLakeEnv(map_name="16x16", is_slippery=True)
    env = gym.make('Taxi-v3')
    print("num states", env.nS)
    env.reset()

    my_policy = MonteCarloExploringStarts(env, gamma=0.9)
    my_policy.monte_carlo_exploring_starts(num_episodes=10000)
    # print(my_policy.Q)
    # my_policy.play_game()
    my_policy.print_rewards_info(num_episodes=1000)
