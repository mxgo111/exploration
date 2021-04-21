from os import system, name
import time
from time import sleep
from IPython.display import clear_output
import gym
import random
import numpy as np

class PolicyIteration:
    """
    Algorithm for Policy Iteration
    """

    # Initialize policy, environment, value table (V)
    def __init__(self, env, policy=None, gamma=1.0):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = policy if policy else self.create_random_policy()
        self.V = np.zeros(self.num_states)
        self.gamma = gamma

    # Creates initial random policy
    def create_random_policy(self):
        # policy is num_states * num_actions array
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        return policy

    # Updates values for one state
    def update_values(self, state):
        v = 0.0
        for action, action_prob in enumerate(self.policy[state]):
            for state_prob, next_state, reward, end in self.env.P[state][action]:
                v += action_prob * state_prob * (reward + self.gamma * self.V[next_state])
        self.V[state] = v

    # finds the best action given value function
    def update_action_values(self, state):
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for prob, next_state, reward, terminated in self.env.P[state][action]:
                action_values[action] = prob * (reward + self.gamma * self.V[next_state])
        return action_values

    # evaluates policy
    def policy_evaluation(self, theta=1e-9, terms=1e6):
        self.V = np.zeros(self.num_states)
        # ensures that we stop even if we don't converge
        prev_sum = 0
        for i in range(int(terms)):
            for state in range(self.num_states):
                self.update_values(state)
            # check if sum of V's is within theta of previous
            # deviates a little from Sutton-Barto
            v = np.sum(self.V)
            if v - prev_sum < theta:
                return
            else:
                prev_sum = v

    # improves policy
    def policy_improvement(self, terms=1e9):
        evals = 1
        # ensures that we stop even if we don't converge
        for i in range(int(terms)):
            stable = True
            self.policy_evaluation()
            for state in range(self.num_states):
                # find current action in a state and compare with best action
                current_action = np.argmax(self.policy[state])
                action_values = self.update_action_values(state)
                best_action = np.argmax(action_values)

                # actions have changed -> unstable
                if current_action != best_action:
                    stable = False
                    self.policy[state] = np.eye(self.num_actions)[best_action]
                evals += 1

            if stable:
                return

    def play_game(self, display=True):
        self.env.reset()
        episodes = []
        finished = False

        while not finished:
            # display current state
            if display:
                clear_output(True)
                self.env.render()
                sleep(1)

            # find next state
            s = self.env.env.s
            action = np.random.choice(np.arange(self.num_actions), p=self.policy[s])
            state, reward, finished, info = self.env.step(action)

            episodes.append([s, action, reward])

        # display end result
        if display:
            clear_output(True)
            self.env.render()
        return episodes

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            while not finished and num_steps < step_limit:
                s = self.env.env.s
                action = np.random.choice(np.arange(self.num_actions), p=self.policy[s])
                state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        return np.mean(total_rewards), np.var(total_rewards)

    # prints reward information
    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        mean, var = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}")

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', is_slippery=False)
    env.reset()

    my_policy = PolicyIteration(env)
    my_policy.policy_improvement()
    my_policy.play_game()
    my_policy.print_rewards_info(num_episodes=10)
