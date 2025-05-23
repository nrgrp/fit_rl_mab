import os
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sp


def fit_direct(envr, min_beta, max_beta, repeats, solver):
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

    htalphas = []
    htbetas = []
    htvalues = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        t = 0

        acts_idx = np.argmax(acts, axis=-1)
        def fn(params):
            alpha = params[:m]
            beta = params[m:]
            qs = [np.zeros(m)]
            for r_idx, r in enumerate(rews[:-1]):
                qs.append((1 - alpha) * qs[r_idx] + alpha * beta * r)
            qs = np.array(qs)
            log_pi = qs - sp.special.logsumexp(qs, axis=-1, keepdims=True)
            ll = np.sum(log_pi[np.arange(acts_idx.shape[0] - 1), acts_idx[1:]])
            return -ll

        ls = []
        xs = []
        for i in range(repeats):
            alpha_init = np.random.uniform(size=m)
            beta_init = np.random.uniform(min_beta, max_beta, size=m)
            bounds = [(0, 1) for _ in range(m)]
            bounds += [(min_beta, max_beta) for _ in range(m)]

            start_t = time.time()
            prob = sp.optimize.minimize(fn, np.hstack((alpha_init, beta_init)), bounds=bounds, method=solver)
            end_t = time.time()
            t += end_t - start_t

            ls.append(prob.fun)
            xs.append(prob.x)

        htalpha = xs[np.argmin(ls)][:m]
        htbeta = xs[np.argmin(ls)][m:]

        times.append(t)
        htalphas.append(htalpha)
        htbetas.append(htbeta)

        ### evaluate ###
        X = []
        Y = []
        G_hat = np.array([htalpha * (1 - htalpha) ** k * htbeta for k in range(n)]).T
        for t in range(n):
            U = np.zeros((n, m))
            if t > 0:
                U[:t] = rews[:t][::-1]
            x = np.sum(np.multiply(G_hat, U.T), axis=-1)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.sum(np.multiply(X, Y), axis=-1) - sp.special.logsumexp(X, axis=1)))
        htvalues.append(X)

    np.save(os.path.join(output_dir, f'htvalues_direct_{solver_tag}.npy'), htvalues)
    np.save(os.path.join(output_dir, f'htalphas_direct_{solver_tag}.npy'), htalphas)
    np.save(os.path.join(output_dir, f'htbetas_direct_{solver_tag}.npy'), htbetas)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, f'log_direct_{solver_tag}.csv'), index=False)


if __name__ == '__main__':
    solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA', 'COBYQA']
    for solver in solvers:
        print(f'Solving using {solver}...')
        fit_direct(envr='2arm', min_beta=0, max_beta=5, repeats=5, solver=solver)
        fit_direct(envr='10arm', min_beta=5, max_beta=10, repeats=5, solver=solver)
