import numpy as np
import gym
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

class EHQ_new:
    def __init__(self, env, alpha, gamma):
        self.env = env

        not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (non primitive actions)
        nA = env.action_space.n + not_pr_acts
        nS = env.observation_space.n
        self.QV = np.zeros((nA, nS, nA))
        self.QC = np.zeros((nA, nS, nA))

        s = self.south = 0
        n = self.north = 1
        e = self.east = 2
        w = self.west = 3
        pickup = self.pickup = 4
        dropoff = self.dropoff = 5
        gotoS = self.gotoS = 6
        gotoD = self.gotoD = 7
        get = self.get = 8
        put = self.put = 9
        root = self.root = 10

        self.graph = [
            set(),  # south
            set(),  # north
            set(),  # east
            set(),  # west
            set(),  # pickup
            set(),  # dropoff
            {s, n, e, w},  # gotoSource
            {s, n, e, w},  # gotoDestination
            {pickup, gotoS},  # get -> pickup, gotoSource
            {dropoff, gotoD},  # put -> dropoff, gotoDestination
            {put, get},  # root -> put, get
        ]

        # record exit states, for subsidies
        self.exit_states = [[set() for _ in range(nA)] for _ in range(nS)]
        self.subsidies = [[dict() for _ in range(nA)] for _ in range(nS)]

        self.gamma = gamma
        self.r_sum = 0
        self.done = False
        self.num_of_ac = 0
        self.alpha = alpha

    def is_primitive(self, act):
        if act <= 5:
            return True
        else:
            return False

    def is_terminal(self, a, done):
        RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
        taxiloc = (taxirow, taxicol)
        if done:
            return True
        elif a == self.root:
            return done
        elif a == self.put:
            return passidx < 4
        elif a == self.get:
            return passidx >= 4
        elif a == self.gotoD:
            return passidx >= 4 and taxiloc == RGBY[destidx]
        elif a == self.gotoS:
            return passidx < 4 and taxiloc == RGBY[passidx]
        elif self.is_primitive(a):
            # just else
            return True

    def findV(self, i, s):
        return max(self.QV[i][s][a] + self.QC[i][s][a] for a in range(11))

    # e-Greedy Approach with eps=0.001
    def greed_act(self, act, s):
#         e = 0.8 ** self.num_of_ac
        e = 0.01
        Q = np.arange(0)
        possible_a = np.arange(0)
        for act2 in self.graph[act]:
            if self.is_primitive(act2) or (not self.is_terminal(act2, self.done)):
                Q = np.concatenate((Q, [self.QV[act, s, act2] + self.QC[act, s, act2]]))
                possible_a = np.concatenate((possible_a, [act2]))

        max_args = np.argwhere(Q == np.amax(Q))
        max_arg = np.random.choice([arg[0] for arg in max_args])

        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    # set subsidies for parent node i, child node a
    # iterates over a's exit states, updates self.subsidies[i][exit_state]
    def set_subsidies(self, i, a):
        if self.exit_states[self.env.s][a]:
            # print(len(self.exit_states[a]))
            min_value = min(self.findV(i, e) for e in self.exit_states[self.env.s][a])
            for exit_state in self.exit_states[self.env.s][a]:
                self.subsidies[self.env.s][i][exit_state] = self.findV(i, exit_state) - min_value

    def EHQ(self, i, parent):  # i is action number
        total_steps = 0
        N = None
#         print("ACTION", i)
        while not self.is_terminal(i, self.done):
            a = self.greed_act(i, self.env.s)
#             print(a)
            r = 0
            s = self.env.s
            if self.is_primitive(a):
                new_s, r, self.done, _ = self.env.step(a)
                self.r_sum += r
                self.num_of_ac += 1
                N = 1
#                 print("ACTION MOVE", a)
            else:
                self.set_subsidies(i, a) # set subsidies for child a from parent i
                bid = self.findV(a, self.env.s)
                N, new_s = self.EHQ(a, i)
                # print("ADF")
                assert(new_s == self.env.s)
                # if new state has a subsidy (i.e. parent knew about such exit state)
                if new_s in self.subsidies[i]:
                    r = bid - (gamma ** N) * self.subsidies[s][i][new_s]
                else:
                    self.exit_states[s][i].add(new_s)
                    r = bid
                    # print("This should never happen")
            # if terminal condition satisfied and not root, then add subsidy
            if self.is_terminal(i, self.done) and parent and (new_s in self.subsidies[s][parent]):
                r += (gamma ** N) * self.subsidies[s][parent][new_s]
#                 if i == self.gotoS:
#                     print("EX ST")
#                     print(list(self.env.decode(new_s)))
#             print(r)

            self.QV[i, s, a] = (1 - self.alpha) * self.QV[i, s, a] + self.alpha * r
            self.QC[i, s, a] = (1 - self.alpha) * self.QC[i, s, a] + self.alpha * (gamma ** N) * self.findV(i, new_s)
#             print("QV--------------------------")
#             print(self.QV[i,s,a])
#             print("QC--------------------------")
#             print(self.QC[i,s,a])
            total_steps += N

        # add to exit states
        self.exit_states[self.env.s][i].add(self.env.s)
#         if i == self.gotoS:
#             print("EX ST bruh")
#             print(list(self.env.decode(new_s)))
#             print("HERE EXIT STATES")
#             for exit in self.exit_states[i]:
#                 print(list(self.env.decode(exit)))
#             print("END HERE")

        return total_steps, new_s

    def reset(self):
        self.env.reset()
        self.r_sum = 0
        self.num_of_ac = 0
        self.done = False

        # reset exit states, subsidies, QV, QC
        # self.exit_states = [set() for _ in range(len(self.exit_states))]
        # self.subsidies = [dict() for _ in range(len(self.subsidies))]
        # self.QV = np.zeros(self.QV.shape)
        # self.QC = np.zeros(self.QC.shape)


alpha = 0.1
gamma = 0.9
env = gym.make('Taxi-v3').env
taxi = EHQ_new(env, alpha, gamma)
episodes = 300
ehq_sum_list = []

for j in range(episodes):
    taxi.reset()
    taxi.EHQ(10, {})      # start in root
    ehq_sum_list.append(taxi.r_sum)
    if (j % 100 == 0):
        print('already made', j, 'episodes')

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 7.5))
plt.plot(ehq_sum_list)
plt.xlabel('episode num')
plt.ylabel('points')
plt.savefig("EHQ")
plt.show()
