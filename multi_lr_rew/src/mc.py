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


def fit_mc(envr, draws, tune, min_beta, max_beta):
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

    htalphas = []
    htbetas = []
    htvalues = []
    times = []
    lls = []
    for r_idx, (rews, acts) in enumerate(zip(rewards, actions)):
        acts_pt = pt.as_tensor_variable(np.argmax(acts, axis=1), dtype='int32')
        rews_pt = pt.as_tensor_variable(rews[:-1], dtype='float32')

        def get_ll(alpha):
            qs = pt.zeros((m,), dtype='float64')
            qs, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[rews_pt], outputs_info=[qs], non_sequences=[alpha]
            )
            log_pi = qs - pt.logsumexp(qs, axis=-1, keepdims=True)
            ll = pt.sum(log_pi[pt.arange(acts_pt.shape[0] - 1), acts_pt[1:]])
            return ll

        with pm.Model() as model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, size=m)
            beta = pm.Uniform(name="beta", lower=min_beta, upper=max_beta, size=m)
            rews_pt *= beta

            like = pm.Potential(name="like", var=get_ll(alpha))

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

        htalpha = mc_log['mean'][:m].to_numpy()
        htbeta = mc_log['mean'][m:].to_numpy()
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

    np.save(os.path.join(output_dir, 'htvalues_mc.npy'), htvalues)
    np.save(os.path.join(output_dir, 'htalphas_mc.npy'), htalphas)
    np.save(os.path.join(output_dir, 'htbetas_mc.npy'), htbetas)
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_mc.csv'), index=False)


if __name__ == '__main__':
    fit_mc(envr='2arm', draws=5000, tune=2000, min_beta=0, max_beta=5)
    fit_mc(envr='10arm', draws=5000, tune=2000, min_beta=5, max_beta=10)
