import os

import numpy as np
import scipy as sp

from utils import Bandit

def get_data(envr, repeats, num_steps, p_r, p_tr, min_beta, max_beta):

    output_dir = f'../data/{envr}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    envr = Bandit(p_r, p_tr)

    alphas = []
    betas = []
    rewards = []  # (repeat, step, reward)
    actions = []  # (repeat, step, action)
    values = []  # (repeat, step, value)
    lls = []
    for rp in range(repeats):
        alpha = np.random.uniform()
        beta = np.random.uniform(min_beta, max_beta)
        alphas.append(alpha)
        betas.append(beta)

        reward, action, value = envr.generate_data(alpha, beta, num_steps)
        action_vec = []
        for a in action:
            a_vec = np.zeros(envr.num_arms)
            a_vec[a] = 1
            action_vec.append(a_vec)

        rewards.append(reward)
        actions.append(action_vec)
        values.append(value)
        lls.append(np.sum(np.sum(np.multiply(value, action_vec), axis=-1) - sp.special.logsumexp(value, axis=1)))
    np.save(os.path.join(output_dir, 'alphas.npy'), alphas)
    np.save(os.path.join(output_dir, 'betas.npy'), betas)
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards)
    np.save(os.path.join(output_dir, 'actions.npy'), actions)
    np.save(os.path.join(output_dir, 'values.npy'), values)
    np.save(os.path.join(output_dir, 'lls.npy'), lls)


if __name__ == '__main__':
    np.random.seed(10015)

    get_data(envr='2arm', repeats=1000, num_steps=200,
             p_r=np.array([0.9, 0.1]), p_tr=0.02, min_beta=0, max_beta=5)
    get_data(envr='10arm', repeats=1000, num_steps=200,
             p_r=np.array([0.30, 0.27, 0.95, 0.67, 0.69, 0.29, 0.42, 0.05, 0.73, 1.00]),
             p_tr=0, min_beta=5, max_beta=10)