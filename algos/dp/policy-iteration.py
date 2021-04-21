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
        if policy:
            self.policy = policy
        else:
            self.policy = self.create_random_policy()
        self.V = np.zeros(self.num_states)
        self.gamma = gamma

    # Creates initial random policy
    def create_random_policy(self):
        # policy is of the form:
        # {
        #   s_0 : { a_0 : pi(a_0|s_0), a_1 : pi(a_1|s_0), ..., a_m : pi(a_m|s_0) }
        #   ...
        #   s_n : { a_0 : pi(a_0|s_n), a_1 : pi(a_1|s_n), ..., a_m : pi(a_m|s_n) }
        # }
        policy = {}
        for key in range(0, self.num_states):
            p = {}
            for action in range(0, self.num_actions):
                p[action] = 1 / self.num_actions
            policy[key] = p
        return policy

    # updates values for one state
    def update_values(self, state):
        v = 0
        for action, action_prob in self.policy[state].items():
            for state_prob, next_state, reward, end in self.env.P[state][action]:
                v += action_prob * state_prob * (reward + self.gamma * self.V[next_state])
                self.V[state] = v

    # def evaluate(V, action_values, env, gamma, state):
    #     for action in range(env.nA):
    #         for prob, next_state, reward, terminated in env.P[state][action]:
    #             action_values[action] += prob * (reward + gamma * V[next_state])
    #     return action_values
    #
    # def lookahead(env, state, V, gamma):
    #     action_values = np.zeros(env.nA)
    #     return evaluate(V, action_values, env, gamma, state)

    def update_best_actions(self, state):
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for prob, next_state, reward, terminated in self.env.P[state][action]:
                action_values[action] = prob * (reward + self.gamma * self.V[next_state])
        return action_values

    # Policy Evaluation
    # def eval_policy(gamma=1.0, theta=1e-9, terms=1e9):
    #     V = np.zeros(self.env.nS)
    #     delta = 0
    #     for i in range(int(terms)):
    #         for state in range(env.nS):
    #             act(V, env, gamma, policy, state, v=0.0)
    #         v = np.sum(V)
    #         if v - delta < theta:
    #             return V
    #         else:
    #             delta = v
    #     return V

    # Policy Improvement
    # def policy_improvement(env, gamma=1.0, terms=1e9):
    #     policy = np.ones([env.nS, env.nA]) / env.nA
    #     evals = 1
    #     for i in range(int(terms)):
    #         stable = True
    #         V = eval_policy(policy, env, gamma=gamma)
    #         for state in range(env.nS):
    #             current_action = np.argmax(policy[state])
    #             action_value = lookahead(env, state, V, gamma)
    #             best_action = np.argmax(action_value)
    #             if current_action != best_action:
    #                 stable = False
    #                 policy[state] = np.eye(env.nA)[best_action]
    #             evals += 1
    #             if stable:
    #                 return policy, V

    def policy_evaluation(self, theta=100, terms=1e6):
        # ensures that we stop even if we don't converge
        for i in range(int(terms)):
            prev_sum = 0
            for state in range(self.num_states):
                self.update_values(state)
            # check if sum of V's is within theta of previous
            # deviates a little from Sutton-Barto
            v = np.sum(self.V)
            if v - prev_sum < theta:
                print("Convergence of V")
                return
            else:
                prev_sum = v

    def policy_improvement(self, terms=1e9):
        policy = np.ones([env.nS, env.nA]) / env.nA
        evals = 1
        # ensures that we stop even if we don't converge
        for i in range(int(terms)):
            stable = True
            self.policy_evaluation()
            for state in range(self.num_states):
                current_action = max(self.policy[state], key=self.policy[state].get)
                action_values = self.update_best_actions(state)
                best_action = np.argmax(action_values)

                # actions have changed -> unstable
                if current_action != best_action:
                    stable = False
                    for action in policy[state]:
                        self.policy[state][action] = 0
                    self.policy[state][best_action] = 1
                evals += 1

            if stable:
                return

    def play_game(self, display=True):
        self.env.reset()
        episodes = []
        finished = False

        while not finished:
            s = self.env.env.s
            if display:
                clear_output(True)
                self.env.render()
                # sleep(1)

            timestep = []
            timestep.append(s)
            # n = random.uniform(0, sum(policy[s]))
            # top_range = 0
            # action = 0
            # for i, prob in enumerate(policy[s]):
            #     top_range += prob
            #     if n < top_range:
            #         action = i
            #         break
            action = random.choices(self.policy[s].keys(), weights=self.policy[s].values())

            state, reward, finished, info = self.env.step(action)

            timestep.append(action)
            timestep.append(reward)

            episodes.append(timestep)

        if display:
            clear_output(True)
            self.env.render()
    #         sleep(0.1)
        return episodes

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env.reset()

    my_policy = PolicyIteration(env)
    my_policy.policy_improvement()
    my_policy.play_game()

# policy, V = improve_policy(env.env)
# play_game(env, policy)

# print(policy, V)
