import os

import numpy as np
import scipy as sp

from utils import Bandit


def get_data(envr, repeats, num_steps, p_r, p_tr, min_beta_r, max_beta_r, min_beta_a, max_beta_a):
    m = p_r.shape[0]

    output_dir = f'../data/{envr}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    envr = Bandit(p_r, p_tr)

    alpha_rs = []
    beta_rs = []
    alpha_as = []
    beta_as = []
    rewards = []  # (repeat, step, reward)
    actions = []  # (repeat, step, action)
    value_rs = []  # (repeat, step, value)
    value_as = []
    lls = []
    for rp in range(repeats):
        alpha_r = np.random.uniform(size=m)
        beta_r = np.random.uniform(min_beta_r, max_beta_r, size=m)
        alpha_a = np.random.uniform(size=m)
        beta_a = np.random.uniform(min_beta_a, max_beta_a, size=m)
        alpha_rs.append(alpha_r)
        beta_rs.append(beta_r)
        alpha_as.append(alpha_a)
        beta_as.append(beta_a)

        reward, action, value_r, value_a = envr.generate_data(alpha_r, beta_r, alpha_a, beta_a, num_steps)
        action_vec = []
        for a in action:
            a_vec = np.zeros(envr.num_arms)
            a_vec[a] = 1
            action_vec.append(a_vec)

        rewards.append(reward)
        actions.append(action_vec)
        value_rs.append(value_r)
        value_as.append(value_a)
        value = value_r + value_a
        lls.append(np.sum(np.sum(np.multiply(value, action_vec), axis=-1) - sp.special.logsumexp(value, axis=1)))
    np.save(os.path.join(output_dir, 'alpha_rs.npy'), alpha_rs)
    np.save(os.path.join(output_dir, 'beta_rs.npy'), beta_rs)
    np.save(os.path.join(output_dir, 'alpha_as.npy'), alpha_as)
    np.save(os.path.join(output_dir, 'beta_as.npy'), beta_as)
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards)
    np.save(os.path.join(output_dir, 'actions.npy'), actions)
    np.save(os.path.join(output_dir, 'value_rs.npy'), value_rs)
    np.save(os.path.join(output_dir, 'value_as.npy'), value_as)
    np.save(os.path.join(output_dir, 'lls.npy'), lls)

if __name__ == '__main__':
    np.random.seed(10015)

    get_data(envr='2arm', repeats=1000, num_steps=200,
             p_r=np.array([0.9, 0.1]), p_tr=0.02, min_beta_r=0, max_beta_r=5,
             min_beta_a=0, max_beta_a=2)
    get_data(envr='10arm', repeats=1000, num_steps=200,
             p_r=np.array([0.30, 0.27, 0.95, 0.67, 0.69, 0.29, 0.42, 0.05, 0.73, 1.00]),
             p_tr=0, min_beta_r=5, max_beta_r=10, min_beta_a=0, max_beta_a=5)
