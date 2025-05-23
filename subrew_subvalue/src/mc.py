import os
import time

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az


def fit_mc(envr, draws, tune, min_beta_r, max_beta_r, min_beta_a, max_beta_a):
    data_dir = f'../data/{envr}'
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    mclog_df = pd.DataFrame()
    log_df = pd.DataFrame()
    output_dir = f'../outputs/{envr}'
    mclog_dir = os.path.join(output_dir, 'mc_val')
    fig_dir = os.path.join(mclog_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    htalpha_rs = []
    htalpha_as = []
    htbeta_rs = []
    htbeta_as = []
    htvalue_rs = []
    htvalue_as = []
    times = []
    lls = []
    for r_idx, (rews, acts) in enumerate(zip(rewards, actions)):
        ys_pt = pt.as_tensor_variable(np.argmax(acts, axis=1), dtype='int32')
        acts_pt = pt.as_tensor_variable(acts[:-1], dtype='int32')
        rews_pt = pt.as_tensor_variable(rews[:-1], dtype='float32')

        def get_ll(alpha_r, alpha_a):
            q_rs = pt.zeros((m,), dtype='float64')
            q_rs, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[rews_pt], outputs_info=[q_rs], non_sequences=[alpha_r]
            )
            q_as = pt.zeros((m,), dtype='float64')
            q_as, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[acts_pt], outputs_info=[q_as], non_sequences=[alpha_a]
            )
            qs = q_rs + q_as
            log_pi = qs - pt.logsumexp(qs, axis=-1, keepdims=True)
            ll = pt.sum(log_pi[pt.arange(ys_pt.shape[0] - 1), ys_pt[1:]])
            return ll

        with pm.Model() as model:
            alpha_r = pm.Uniform(name="alpha_r", lower=0, upper=1, size=m)
            beta_r = pm.Uniform(name="beta_r", lower=min_beta_r, upper=max_beta_r, size=m)
            alpha_a = pm.Uniform(name="alpha_a", lower=0, upper=1, size=m)
            beta_a = pm.Uniform(name="beta_a", lower=min_beta_a, upper=max_beta_a, size=m)
            rews_pt *= beta_r
            acts_pt *= beta_a

            like = pm.Potential(name="like", var=get_ll(alpha_r, alpha_a))

            start_time = time.time()
            tr = pm.sample(draws=draws, tune=tune, nuts_sampler='numpyro')
            end_time = time.time()
        times.append(end_time - start_time)

        mc_log = az.summary(tr)
        mc_log['repeat'] = r_idx
        mc_log = mc_log.reset_index(names='param')
        mclog_df = pd.concat((mclog_df, mc_log))
        mclog_df.to_csv(os.path.join(mclog_dir, 'mc_log.csv'), index=False)

        az.plot_trace(tr)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'trace_{r_idx}.png'))
        plt.close()

        az.plot_posterior(tr)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'posterior_{r_idx}.png'))
        plt.close()

        mc_log = mc_log.set_index('param')
        htalpha_r = np.array([mc_log.loc[f'alpha_r[{k}]', 'mean'] for k in range(m)])
        htbeta_r = np.array([mc_log.loc[f'beta_r[{k}]', 'mean'] for k in range(m)])
        htalpha_a = np.array([mc_log.loc[f'alpha_a[{k}]', 'mean'] for k in range(m)])
        htbeta_a = np.array([mc_log.loc[f'beta_a[{k}]', 'mean'] for k in range(m)])
        htalpha_rs.append(htalpha_r)
        htbeta_rs.append(htbeta_r)
        htalpha_as.append(htalpha_a)
        htbeta_as.append(htbeta_a)

        ### evaluate ###
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
            X.append(x_r + x_a)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.sum(np.multiply(X, Y), axis=-1) - sp.special.logsumexp(X, axis=1)))
        htvalue_rs.append(x_r)
        htvalue_as.append(x_a)

    np.save(os.path.join(output_dir, 'htvalue_rs_mc.npy'), htvalue_rs)
    np.save(os.path.join(output_dir, 'htvalue_as_mc.npy'), htvalue_as)
    np.save(os.path.join(output_dir, 'htalpha_rs_mc.npy'), htalpha_rs)
    np.save(os.path.join(output_dir, 'htalpha_as_mc.npy'), htalpha_as)
    np.save(os.path.join(output_dir, 'htbeta_rs_mc.npy'), htbeta_rs)
    np.save(os.path.join(output_dir, 'htbeta_as_mc.npy'), htbeta_as)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_mc.csv'), index=False)


if __name__ == '__main__':
    fit_mc(envr='2arm', draws=5000, tune=2000, min_beta_r=0, max_beta_r=5,
           min_beta_a=0, max_beta_a=2)
    fit_mc(envr='10arm', draws=5000, tune=2000, min_beta_r=5, max_beta_r=10,
           min_beta_a=0, max_beta_a=5)
