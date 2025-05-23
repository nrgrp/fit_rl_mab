import copy
import numpy as np


class Bandit:
    def __init__(self, p_r, p_tr):
        self.p_r = p_r
        self.num_arms = p_r.shape[0]
        self.p_tr = p_tr

    def generate_data(self, alpha, beta, num_steps):
        rewards = []
        actions = []
        values = []
        v = np.ones(self.num_arms)
        for _ in range(num_steps):
            pi = np.exp(v) / np.sum(np.exp(v), keepdims=True)
            a = np.random.choice(self.num_arms, p=pi)
            r = np.zeros(self.num_arms)
            if np.random.uniform() < self.p_r[a]:
                r[a] += 1

            rewards.append(copy.deepcopy(r))
            actions.append(a)
            values.append(v)

            r *= beta
            v = (1 - alpha) * v + alpha * r

            if np.random.uniform() < self.p_tr:
                np.random.shuffle(self.p_r)
        return np.array(rewards), np.array(actions), np.array(values)
