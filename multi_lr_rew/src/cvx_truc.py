import os
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import cvxpy as cp
import scipy as sp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def fit_relax_truc(envr, p):
    data_dir = f'../data/{envr}'
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = f'../outputs/{envr}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    htvalues = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        ### step I ###
        G = cp.Variable((m, p))

        X = []
        Y = []
        for t in range(n):
            if t < p:
                U = np.zeros((p, m))
                U[:t] = rews[:t][::-1]
            else:
                U = rews[t - p: t][::-1]
            x = cp.sum(cp.multiply(G, U.T), axis=1)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = cp.vstack(X)
        obj = cp.sum(cp.sum(cp.multiply(X, Y), axis=1) - cp.log_sum_exp(X, axis=1))
        constraints = []
        constraints.append(G[:, -1] >= 0)
        constraints.append(cp.diff(G, axis=1) <= 0)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        assert prob.is_dcp()
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(reduced_tol_gap_abs=1e-3, reduced_tol_gap_rel=1e-3)
        times.append(prob.solver_stats.solve_time)

        ### evaluate ###
        X = []
        Y = []
        G_hat = G.value
        for t in range(n):
            if t < p:
                U = np.zeros((p, m))
                U[:t] = rews[:t][::-1]
            else:
                U = rews[t - p: t][::-1]
            x = np.sum(np.multiply(G_hat, U.T), axis=-1)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.sum(np.multiply(X, Y), axis=-1) - sp.special.logsumexp(X, axis=1)))
        htvalues.append(X)

    np.save(os.path.join(output_dir, f'htvalues_cvx_truc.npy'), htvalues)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, f'log_cvx_truc.csv'), index=False)


if __name__ == '__main__':
    # fit_relax_truc(envr='2arm', p=5)
    # fit_relax_truc(envr='10arm', p=5)

    solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA', 'COBYQA']
    args = [['2arm', 5, 0, 5, 5], ['10arm', 5, 5, 10, 5]]
    for s2_solver in solvers:
        solver_tag = ''.join(s2_solver.split('-')).lower()
        print(f'Solving using {s2_solver}...')
        for envr, p, min_beta, max_beta, s2_repeats in args:
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
            s1_times = []
            s2_times = []
            s1_lls = []
            lls = []
            for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
                ### step I ###
                G = cp.Variable((m, p))

                X = []
                Y = []
                for t in range(n):
                    if t < p:
                        U = np.zeros((p, m))
                        U[:t] = rews[:t][::-1]
                    else:
                        U = rews[t - p: t][::-1]
                    x = cp.sum(cp.multiply(G, U.T), axis=1)
                    X.append(x)
                    Y.append(acts[t])
                Y = np.vstack(Y)
                X = cp.vstack(X)
                obj = cp.sum(cp.sum(cp.multiply(X, Y), axis=1) - cp.log_sum_exp(X, axis=1))
                constraints = []
                constraints.append(G[:, -1] >= 0)
                constraints.append(cp.diff(G, axis=1) <= 0)

                s1_prob = cp.Problem(cp.Maximize(obj), constraints)
                assert s1_prob.is_dcp()
                s1_prob.solve()
                s1_times.append(s1_prob.solver_stats.solve_time)
                s1_lls.append(s1_prob.value)

                ### step II ###
                def solve_s2(y):
                    def fn(params):
                        alpha, beta = params
                        return np.sum([(alpha * (1 - alpha) ** k * beta - y[k]) ** 2 for k in range(p)])

                    s2_ls = []
                    s2_xs = []
                    s2_t = 0
                    for _ in range(s2_repeats):
                        alpha_init = np.random.uniform()
                        beta_init = np.random.uniform(min_beta, max_beta)
                        bounds = [(0, 1), (min_beta, max_beta)]

                        start_t = time.time()
                        s2_prob = sp.optimize.minimize(fn, (alpha_init, beta_init), bounds=bounds, method=s2_solver)
                        end_t = time.time()
                        s2_t += end_t - start_t

                        loss = s2_prob.fun
                        s2_ls.append(loss)
                        s2_xs.append(s2_prob.x)

                    best_idx = np.argmin(s2_ls)
                    best_alpha, best_beta = s2_xs[best_idx]
                    return best_alpha, best_beta, s2_t

                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(solve_s2, G.value))
                htalpha, htbeta, s2_t = np.array(results).T

                s2_times.append(np.max(s2_t))
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

            np.save(os.path.join(output_dir, f'htvalues_cvx_truc_{solver_tag}.npy'), htvalues)
            np.save(os.path.join(output_dir, f'htalphas_cvx_truc_{solver_tag}.npy'), htalphas)
            np.save(os.path.join(output_dir, f'htbetas_cvx_truc_{solver_tag}.npy'), htbetas)
            log_df['s1_time'] = s1_times
            log_df['s2_time'] = s2_times
            log_df['s1_ll'] = s1_lls
            log_df['ll'] = lls
            log_df.to_csv(os.path.join(output_dir, f'log_cvx_truc_{solver_tag}.csv'), index=False)
