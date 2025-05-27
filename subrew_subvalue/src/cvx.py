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


def fit_relax(envr):
    data_dir = f'../data/{envr}'
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = f'../outputs/{envr}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    htvalue_rs = []
    htvalue_as = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        ### step I ###
        G_r = cp.Variable((m, n))
        G_a = cp.Variable((m, n))

        X = []
        Y = []
        for t in range(n):
            U_r = np.zeros((n, m))
            U_a = np.zeros((n, m))
            if t > 0:
                U_r[:t] = rews[:t][::-1]
                U_a[:t] = acts[:t][::-1]
            x = cp.sum(cp.multiply(G_r, U_r.T), axis=1) + cp.sum(cp.multiply(G_a, U_a.T), axis=1)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = cp.vstack(X)
        obj = cp.sum(cp.sum(cp.multiply(X, Y), axis=1) - cp.log_sum_exp(X, axis=1))
        constraints = []
        constraints.append(G_r[:, -1] >= 0)
        constraints.append(cp.diff(G_r, axis=1) <= 0)
        constraints.append(G_a[:, -1] >= 0)
        constraints.append(cp.diff(G_a, axis=1) <= 0)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        assert prob.is_dcp()
        try:
            prob.solve()
        except cp.SolverError:
            if envr == '2arm':
                prob.solve(reduced_tol_gap_abs=1e-2, reduced_tol_gap_rel=1e-2)
            elif envr == '10arm':
                prob.solve(reduced_tol_gap_abs=1e-1, reduced_tol_gap_rel=1e-1)
        times.append(prob.solver_stats.solve_time)

        ### evaluate ###
        X_r = []
        X_a = []
        X = []
        Y = []
        G_r_hat = G_r.value
        G_a_hat = G_a.value
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

    np.save(os.path.join(output_dir, f'htvalue_rs_cvx.npy'), htvalue_rs)
    np.save(os.path.join(output_dir, f'htvalue_as_cvx.npy'), htvalue_as)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, f'log_cvx.csv'), index=False)


if __name__ == '__main__':
    # fit_relax(envr='2arm')
    # fit_relax(envr='10arm')

    solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA', 'COBYQA']
    args = [['2arm', 0, 5, 0, 2, 5], ['10arm', 5, 10, 0, 5, 5]]
    for s2_solver in solvers:
        solver_tag = ''.join(s2_solver.split('-')).lower()
        print(f'Solving using {s2_solver}...')
        for envr, min_beta_r, max_beta_r, min_beta_a, max_beta_a, s2_repeats in args:
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
            s1_times = []
            s2_times = []
            s1_lls = []
            lls = []
            for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
                ### step I ###
                G_r = cp.Variable((m, n))
                G_a = cp.Variable((m, n))

                X = []
                Y = []
                for t in range(n):
                    U_r = np.zeros((n, m))
                    U_a = np.zeros((n, m))
                    if t > 0:
                        U_r[:t] = rews[:t][::-1]
                        U_a[:t] = acts[:t][::-1]
                    x = cp.sum(cp.multiply(G_r, U_r.T), axis=1) + cp.sum(cp.multiply(G_a, U_a.T), axis=1)
                    X.append(x)
                    Y.append(acts[t])
                Y = np.vstack(Y)
                X = cp.vstack(X)
                obj = cp.sum(cp.sum(cp.multiply(X, Y), axis=1) - cp.log_sum_exp(X, axis=1))
                constraints = []
                constraints.append(G_r[:, -1] >= 0)
                constraints.append(cp.diff(G_r, axis=1) <= 0)
                constraints.append(G_a[:, -1] >= 0)
                constraints.append(cp.diff(G_a, axis=1) <= 0)

                s1_prob = cp.Problem(cp.Maximize(obj), constraints)
                assert s1_prob.is_dcp()
                try:
                    s1_prob.solve()
                except cp.SolverError:
                    if envr == '2arm':
                        s1_prob.solve(reduced_tol_gap_abs=1e-2, reduced_tol_gap_rel=1e-2)
                    elif envr == '10arm':
                        s1_prob.solve(reduced_tol_gap_abs=1e-1, reduced_tol_gap_rel=1e-1)
                    else:
                        raise NotImplementedError
                s1_times.append(s1_prob.solver_stats.solve_time)
                s1_lls.append(s1_prob.value)

                ### step II ###
                def solve_s2(y, min_beta, max_beta):
                    def fn(params):
                        alpha, beta = params
                        return np.sum([(alpha * (1 - alpha) ** k * beta - y[k]) ** 2 for k in range(n)])

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

                Gvalues = np.vstack((G_r.value, G_a.value))
                min_betas = np.hstack((np.repeat(min_beta_r, m), np.repeat(min_beta_a, m)))
                max_betas = np.hstack((np.repeat(max_beta_r, m), np.repeat(max_beta_a, m)))
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(solve_s2, Gvalues, min_betas, max_betas))
                htalpha, htbeta, s2_t = np.array(results).T
                htalpha_r = htalpha[:m]
                htalpha_a = htalpha[m:]
                htbeta_r = htbeta[:m]
                htbeta_a = htbeta[m:]

                s2_times.append(np.max(s2_t))
                htalpha_rs.append(htalpha_r)
                htbeta_rs.append(htbeta_r)
                htalpha_as.append(htalpha_a)
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

            np.save(os.path.join(output_dir, f'htvalue_rs_cvx_{solver_tag}.npy'), htvalue_rs)
            np.save(os.path.join(output_dir, f'htvalue_as_cvx_{solver_tag}.npy'), htvalue_as)
            np.save(os.path.join(output_dir, f'htalpha_rs_cvx_{solver_tag}.npy'), htalpha_rs)
            np.save(os.path.join(output_dir, f'htalpha_as_cvx_{solver_tag}.npy'), htalpha_as)
            np.save(os.path.join(output_dir, f'htbeta_rs_cvx_{solver_tag}.npy'), htbeta_rs)
            np.save(os.path.join(output_dir, f'htbeta_as_cvx_{solver_tag}.npy'), htbeta_as)
            log_df['s1_time'] = s1_times
            log_df['s2_time'] = s2_times
            log_df['s1_ll'] = s1_lls
            log_df['ll'] = lls
            log_df.to_csv(os.path.join(output_dir, f'log_cvx_{solver_tag}.csv'), index=False)