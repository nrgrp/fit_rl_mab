import os
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sp


def fit_direct(envr, min_beta_r, max_beta_r, min_beta_a, max_beta_a, repeats, solver):
    solver_tag = ''.join(solver.split('-')).lower()

    data_dir = f'../data/{envr}'
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = f'../outputs/{envr}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    htalpha_rs = []
    htalpha_as = []
    htbeta_rs = []
    htbeta_as = []
    htvalue_rs = []
    htvalue_as = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        t = 0

        acts_idx = np.argmax(acts, axis=-1)
        def fn(params):
            alpha_r = params[:m]
            alpha_a = params[m: 2 * m]
            beta_r = params[2 * m: 3 * m]
            beta_a = params[-m:]
            q_rs = [np.zeros(m)]
            q_as = [np.zeros(m)]
            for idx, (r, a) in enumerate(zip(rews[:-1], acts[:-1])):
                q_rs.append((1 - alpha_r) * q_rs[idx] + alpha_r * beta_r * r)
                q_as.append((1 - alpha_a) * q_as[idx] + alpha_a * beta_a * a)

            q_rs = np.array(q_rs)
            q_as = np.array(q_as)
            qs = q_rs + q_as
            log_pi = qs - sp.special.logsumexp(qs, axis=-1, keepdims=True)
            ll = np.sum(log_pi[np.arange(acts_idx.shape[0] - 1), acts_idx[1:]])
            return -ll

        ls = []
        xs = []
        for i in range(repeats):
            alpha_init = np.random.uniform(size=2 * m)
            beta_init = np.hstack((
                np.random.uniform(min_beta_r, max_beta_r, size=m),
                np.random.uniform(min_beta_a, max_beta_a, size=m),
            ))
            bounds = [(0, 1) for _ in range(2 * m)]
            bounds += [(min_beta_r, max_beta_r) for _ in range(m)]
            bounds += [(min_beta_a, max_beta_a) for _ in range(m)]

            start_t = time.time()
            prob = sp.optimize.minimize(fn, np.hstack((alpha_init, beta_init)), bounds=bounds, method=solver)
            end_t = time.time()
            t += end_t - start_t

            ls.append(prob.fun)
            xs.append(prob.x)

        htalpha_r = xs[np.argmin(ls)][:m]
        htalpha_a = xs[np.argmin(ls)][m: 2 * m]
        htbeta_r = xs[np.argmin(ls)][2 * m: 3 * m]
        htbeta_a = xs[np.argmin(ls)][-m:]

        times.append(t)
        htalpha_rs.append(htalpha_r)
        htalpha_as.append(htalpha_a)
        htbeta_rs.append(htbeta_r)
        htbeta_as.append(htbeta_a)

        ### evaluate ###
        X_r = []
        X_a = []
        X = []
        Y = []
        G_r_hat = np.array([htalpha_r * (1 - htalpha_r) ** k * htbeta_r for k in range(n)]).T
        G_a_hat = np.array([htalpha_a * (1 - htalpha_a) ** k * htbeta_a for k in range(n)]).T

        for t in range(n):
            U_r = np.zeros((n, m))
            U_a = np.zeros((n, m))
            if t > 0:
                U_r[:t] = rews[:t][::-1]
                U_a[:t] = acts[:t][::-1]
            x_r = np.sum(np.multiply(G_r_hat, U_r.T), axis=-1)
            x_a = np.sum(np.multiply(G_a_hat, U_a.T), axis=-1)
            X_r.append(x_r)
            X_a.append(x_a)
            X.append(x_r + x_a)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X_r = np.vstack(X_r)
        X_a = np.vstack(X_a)
        X = np.vstack(X)
        lls.append(np.sum(np.sum(np.multiply(X, Y), axis=-1) - sp.special.logsumexp(X, axis=1)))
        htvalue_rs.append(X_r)
        htvalue_as.append(X_a)

    np.save(os.path.join(output_dir, f'htvalue_rs_direct_{solver_tag}.npy'), htvalue_rs)
    np.save(os.path.join(output_dir, f'htvalue_as_direct_{solver_tag}.npy'), htvalue_as)
    np.save(os.path.join(output_dir, f'htalpha_rs_direct_{solver_tag}.npy'), htalpha_rs)
    np.save(os.path.join(output_dir, f'htalpha_as_direct_{solver_tag}.npy'), htalpha_as)
    np.save(os.path.join(output_dir, f'htbeta_rs_direct_{solver_tag}.npy'), htbeta_rs)
    np.save(os.path.join(output_dir, f'htbeta_as_direct_{solver_tag}.npy'), htbeta_as)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, f'log_direct_{solver_tag}.csv'), index=False)


if __name__ == '__main__':
    solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA', 'COBYQA']
    for solver in solvers:
        print(f'Solving using {solver}...')
        fit_direct(envr='2arm', min_beta_r=0, max_beta_r=5,
                   min_beta_a=0, max_beta_a=2, repeats=5, solver=solver)
        fit_direct(envr='10arm', min_beta_r=5, max_beta_r=10,
                   min_beta_a=0, max_beta_a=5, repeats=5, solver=solver)