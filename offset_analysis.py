import os
import mne
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd

import pymc as pm
import arviz as az
import pytensor.tensor as pt

import environment_variables as ev
from helper_function.helper_general import get_roi_channels
import matplotlib.pyplot as plt
from cog_ieeg import plot_ieeg_image
from mne.baseline import rescale
from mne.stats import permutation_cluster_1samp_test




# Set list of views:
views = ['lateral', 'medial', 'rostral', 'caudal', 'ventral', 'dorsal']
pfc_roi = [
      "ctx_lh_G_and_S_cingul-Ant",
      "ctx_rh_G_and_S_cingul-Ant",
      "ctx_lh_G_and_S_cingul-Mid-Ant",
      "ctx_rh_G_and_S_cingul-Mid-Ant",
      "ctx_lh_G_and_S_cingul-Mid-Post",
      "ctx_rh_G_and_S_cingul-Mid-Post",
      "ctx_lh_G_front_inf-Opercular",
      "ctx_rh_G_front_inf-Opercular",
      "ctx_lh_G_front_inf-Orbital",
      "ctx_rh_G_front_inf-Orbital",
      "ctx_lh_G_front_inf-Triangul",
      "ctx_rh_G_front_inf-Triangul",
      "ctx_lh_G_front_middle",
      "ctx_rh_G_front_middle",
      "ctx_lh_Lat_Fis-ant-Horizont",
      "ctx_rh_Lat_Fis-ant-Horizont",
      "ctx_lh_Lat_Fis-ant-Vertical",
      "ctx_rh_Lat_Fis-ant-Vertical",
      "ctx_lh_S_front_inf",
      "ctx_rh_S_front_inf",
      "ctx_lh_S_front_middle",
      "ctx_rh_S_front_middle",
      "ctx_lh_S_front_sup",
      "ctx_rh_S_front_sup"
]


def mixture_model(
    y, 
    k, 
    draws=1000, 
    tune=1000, 
    chains=4, 
    target_accept=0.95, 
    seed=0, 
    do_zscore=True
):
    """
    Fit gaussian mixture model with n components

    Parameters
    ----------
    y : np.array (n_sample, )
        Data to fit the mixture model to
    k : int
        number of components
    draws : int, optional
        number of draws for the sampler, by default 1000
    tune : int, optional
        number of tuning draws, by default 1000
    chains : int, optional
        number of sampling chains, by default 4
    target_accept : float, optional
        target accept for sampling, by default 0.95
    seed : int, optional
        seed for sampler, by default 0
    do_zscore : bool, optional
        whether to z score the data before model fitting, by default True

    Returns
    -------
    pymc inference data object
        inference data of the fitted model with log likelihood

    Raises
    ------
    ValueError
        The function only handles two component models
    """
    if do_zscore:
        y = zscore(y)
    if k > 2:
        raise ValueError('THe model cannot handle more than 2 components!!')
    if k == 1:
        with pm.Model() as model:
            # Prior over the mean of each process
            mu = pm.Normal("mu", mu=0, sigma=2)
            # Prior over the deviation of each process
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            # Model
            pm.Normal("x", mu=mu, sigma=sigma, observed=y)
            # Sample
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed,
                progressbar=False,
            )
        # Compute the likelihood:
        idata = pm.compute_log_likelihood(idata, model=model)
    else:
        initval = {
            "mu0": 0,
            "delta": 1.0,
        }
        with pm.Model(coords={"cluster": range(k)}) as model:
            # Prior over the mean of each process
                # Base mean
            mu0 = pm.Normal("mu0", mu=0, sigma=2)

            # Positive separation
            delta = pm.HalfNormal("delta", sigma=2.0)

            # Construct ordered means
            mu = pm.Deterministic(
                "mu",
                pm.math.stack([mu0, mu0 + delta]),
                dims="cluster",
            )
            # Prior over the deviation of each process
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="cluster")
            # Prior over the proportion of each processes
            w = pm.Dirichlet("w", np.ones(k) + 1, dims="cluster")
            # Mixture model:
            pm.NormalMixture("x", w=w, mu=mu, sigma=sigma, observed=y)
            # Sample
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,initvals=initval,
                target_accept=target_accept, random_seed=seed,
                progressbar=False,
            )
        # Compute the likelihood:
        idata = pm.compute_log_likelihood(idata, model=model)
    return idata
    

def plot_mixture_fit(
    idata,
    y,
    *,
    bins=40,
    seed=0,
    do_zscore=True,
    scale_label=None,
    title=None,
    n_posterior_draws=200,
    band=(5, 95),
    hist_alpha=0.35,
    band_alpha=0.2,
    comp_alpha=1.0,
):
    """
    Plot posterior mean density and an uncertainty band for either:
      - 1-component Normal model: posterior over (mu, sigma)
      - 2-component NormalMixture: posterior over (w, mu, sigma)

    Parameters
    ----------
    idata : arviz.InferenceData
        Output from your mixture_model function.
    y : array-like
        Raw data (will be z-scored if zscore=True to match your model usage).
    bins : int
        Histogram bins.
    seed : int
        RNG seed for posterior draw subsampling.
    do_zscore : bool
        If True, z-score y before plotting (should match model fit setting).
    scale_label : str or None
        X-axis label. If None, inferred from zscore.
    title : str or None
        Plot title.
    n_posterior_draws : int
        Number of posterior draws to sample for the uncertainty band.
    band : tuple(float, float)
        Percentiles for uncertainty band, e.g. (5,95).
    """

    # Do zcoring if necessary:
    if do_zscore:
        y_plot = zscore(np.asarray(y))
        if scale_label is None:
            scale_label = "z-scored y"
    else:
        y_plot = np.asarray(y)
        if scale_label is None:
            scale_label = "y"

    post = idata.posterior

    # Detect whether this is k=1 or k=2 from available variables
    has_w = "w" in post
    has_mu = "mu" in post
    has_sigma = "sigma" in post
    if not (has_mu and has_sigma):
        raise ValueError("Posterior must contain 'mu' and 'sigma'.")

    # Set x grid
    lo, hi = np.percentile(y_plot, [0.5, 99.5])
    pad = 0.15 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    x = np.linspace(lo, hi, 600)

    # Set seed:
    rng = np.random.default_rng(seed)

    # Plotting:
    fig, ax = plt.subplots(figsize=(7, 4))
    # Plot the data:
    ax.hist(y_plot, bins=bins, density=True, alpha=hist_alpha, edgecolor="none")

    # -------------------------
    # Plot 1-component model
    # -------------------------
    if not has_w:
        # mu, sigma are scalars per draw
        mus = post["mu"].values.reshape(-1)       # (S,)
        sigmas = post["sigma"].values.reshape(-1) # (S,)

        mu_mean = mus.mean()
        s_mean = sigmas.mean()

        # Posterior-mean density (approx via mean params)
        p_mean = norm.pdf(x, mu_mean, s_mean)

        # Uncertainty band from posterior draws
        S = mus.shape[0]
        n = min(n_posterior_draws, S)
        idx = rng.choice(S, size=n, replace=False)

        p_draws = np.zeros((n, x.size), dtype=float)
        for i, s_idx in enumerate(idx):
            p_draws[i] = norm.pdf(x, mus[s_idx], sigmas[s_idx])

        lower = np.percentile(p_draws, band[0], axis=0)
        upper = np.percentile(p_draws, band[1], axis=0)

        ax.fill_between(
            x, lower, upper, alpha=band_alpha,
            label=f"Normal {band[0]}–{band[1]}% band (posterior draws)"
        )
        ax.plot(x, p_mean, linewidth=2.5, label="Normal (posterior mean)")

    # -------------------------
    # Plot 2-component mixture model
    # -------------------------
    else:
        # w, mu, sigma are vectors per draw, length 2
        ws  = post["w"].values.reshape(-1, 2)      # (S,2)
        mus = post["mu"].values.reshape(-1, 2)     # (S,2)
        ss  = post["sigma"].values.reshape(-1, 2)  # (S,2)

        w_mean = ws.mean(axis=0)
        mu_mean = mus.mean(axis=0)
        s_mean = ss.mean(axis=0)

        # Posterior-mean component densities + mixture
        ps = []
        pmix_mean = np.zeros(x.shape, dtype=float)
        for w, m, s in zip(w_mean, mu_mean, s_mean):
            p = norm.pdf(x, m, s)
            ps.append(p)
            pmix_mean += w * p

        # Uncertainty band: mixture density per posterior draw
        S = ws.shape[0]
        n = min(n_posterior_draws, S)
        idx = rng.choice(S, size=n, replace=False)

        pmix_draws = np.zeros((n, x.size), dtype=float)
        for i, s_idx in enumerate(idx):
            # Loop through components:
            for ii in range(ws.shape[-1]):
                # mixture across components
                pmix_draws[i] += ws[s_idx, ii] * norm.pdf(x, mus[s_idx, ii], ss[s_idx, ii])

        lower = np.percentile(pmix_draws, band[0], axis=0)
        upper = np.percentile(pmix_draws, band[1], axis=0)

        ax.fill_between(
            x, lower, upper, alpha=band_alpha,
            label=f"Mixture {band[0]}–{band[1]}% band (posterior draws)"
        )
        for i in range(ws.shape[-1]):
            ax.plot(
                x, w_mean[i] * ps[i],
                linewidth=2, alpha=comp_alpha,
                label=f"Comp {i} (w={w_mean[i]:.2f})"
            )
        ax.plot(x, pmix_mean, linewidth=2.5, label="Mixture (posterior mean)")

    ax.set_xlabel(scale_label)
    ax.set_ylabel("Density")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig, ax


def compare_models_loo(idata0, idata1, print_summary=False):
    """
    Returns (loo0, loo1, comparison_df, wins, delta_elpd, delta_se)
    where wins=True means mixture wins by a conservative margin.
    """
    loo0 = az.loo(idata0, pointwise=True)
    loo1 = az.loo(idata1, pointwise=True)

    # ArviZ compare: higher elpd_loo is better
    cmp = az.compare({"single": idata0, "mixture": idata1}, ic="loo")

    # Compute delta (mixture - single) and its SE from compare table
    # cmp.loc[...] has columns like "elpd_loo" and "se"
    elpd_single = cmp.loc["single", "elpd_loo"]
    elpd_mix = cmp.loc["mixture", "elpd_loo"]
    se_single = cmp.loc["single", "se"]
    se_mix = cmp.loc["mixture", "se"]

    delta_elpd = elpd_mix - elpd_single
    # Conservative SE for delta: sqrt(se1^2 + se0^2)
    delta_se = float(np.sqrt(se_mix**2 + se_single**2))

    # Conservative win rule: delta > 2 * SE
    wins = bool(delta_elpd > 2.0 * delta_se)

    # Return the pareto K:
    max_parto_k = max([max(loo1.pareto_k.values), max(loo0.pareto_k.values)])

    if print_summary:
        print('')
        print('='*40)
        print('Result of one vs. two components model comparison:')
        print(f'   Two components model LOO={cmp.loc["mixture", "elpd_loo"]}')
        print(f'   One components model LOO={cmp.loc["single", "elpd_loo"]}')
        print(f'   Delta elpd={delta_elpd}')
        print(f'   Delta se={delta_se}')
        print(f'   Normalized Delta elpd (> 2 = evidence for 2 comp)={delta_elpd/delta_se}')
        print(f'Two components model wins? {wins}')

    return loo0, loo1, cmp, wins, float(delta_elpd), float(delta_se), max_parto_k


def posterior_trial_membership(idata_mixture, threshold=0.5):
    """
    Extract per-trial P(comp1|y) and hard labels.
    """
    p = idata_mixture.posterior["p_comp1"].values  # (chains, draws, trials)
    p_flat = p.reshape(-1, p.shape[-1])
    p_mean = p_flat.mean(axis=0)
    hard = (p_mean >= threshold).astype(int)
    return p_mean, hard


def robust_z(x, axis=-1, eps=1e-12):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    # 1.4826 * MAD ~= SD for Gaussian
    return (x - med) / (1.4826 * mad + eps)


def compute_bad_amp_per_channel(epochs, tmin=-0.3, tmax=0.6, z_thr=10):
    """Returns:
      bad_amp: (n_epochs, n_ch) boolean
      good_idx_by_ch: dict ch_name -> 1D int array of epoch indices kept for that channel
    """
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = ep.get_data()  # (n_epochs, n_ch, n_times)

    Z = robust_z(X, axis=-1)
    max_z = np.max(np.abs(Z), axis=-1)    # (n_epochs, n_ch)
    bad_amp = max_z > z_thr               # (n_epochs, n_ch)

    good_idx_by_ch = {
        ch: np.where(~bad_amp[:, ci])[0]
        for ci, ch in enumerate(ep.ch_names)
    }
    return bad_amp, good_idx_by_ch, ep

def print_rejection_summary(bad_amp, ch_names):
    n_epochs = bad_amp.shape[0]
    bad_counts = bad_amp.sum(axis=0)
    order = np.argsort(-bad_counts)
    for ci in order:
        c = int(bad_counts[ci])
        if c == 0:
            break
        print(f"{ch_names[ci]}: rejected {c}/{n_epochs} "
              f"({100*c/n_epochs:.1f}%)")


def offset_analysis(subjects, data_root, analysis_name="offset_analysis"):
    """
    Perform offset analysis on iEEG data.

    :param parameters_file: (str) Path to the parameters file in JSON format.
    :param subjects: (list) List of subjects.
    :param data_root: (str) Root directory for data.
    :param analysis_name: (str) Name of the analysis.
    :param task_conditions: (list) List of task conditions.
    :return: None
    """
    # Prepare save directory:
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, 'Dur', 'offset_analysis')
    os.makedirs(save_dir, exist_ok=True)
    tmin, tmax = 0.2, 0.6
    winning_electrodes = []
    # Loop through each subjects:
    for sub in subjects:
        # Load epochs data:
        epochs_file = Path(data_root, "derivatives", "preprocessing", f"sub-{sub}", 
                    "ses-1", 'ieeg', 'epoching', 'high_gamma',
                    f"sub-{sub}_ses-1_task-Dur_desc-epoching_ieeg-epo.fif")
        # Get PFC data:
        picks = get_roi_channels(ev.cog_bids_root, sub, "1", "destrieux", pfc_roi)
        if len(picks) == 0:
            continue
        try:
            epochs = mne.read_epochs(epochs_file).pick(picks).pick(['ecog', 'seeg'],exclude='bads')
        except FileNotFoundError:
            continue      

        # Detect artifacts:
        bad_amp, good_idx_by_ch, _ = compute_bad_amp_per_channel(epochs, tmin=-0.3, tmax=0.6, z_thr=10)
        print_rejection_summary(bad_amp, epochs.ch_names)
        # =============================================================
        # Plot electrodes with activation ranked per activation strength:
        # Loop through each channels and calculate onset mean amplitude:
        amp = []
        keep = []
        fsize = []
        for pick in epochs.ch_names:
            # Remove artifacts:
            pick_epoch = epochs.copy().pick(pick)[good_idx_by_ch[pick]]["stimulus onset/Relevant non-target"]
            # Extract onset data:
            data = np.squeeze(pick_epoch.get_data(picks=pick, tmin=tmin, tmax=tmax))
            # Cluster based permutation test:
                # 1-sample cluster permutation across time
            _, _, cluster_pvals, _ = permutation_cluster_1samp_test(
                data - 1,
                n_permutations=2000,
                tail=0,
                adjacency=None,   # temporal adjacency is handled for 2D (n_samples, n_times)
                out_type="mask",
                n_jobs=1,         # set >1 if you want parallel and your environment supports it
                seed=0,
            )
            # Keep electrode if any time-cluster is significant
            if np.any(cluster_pvals < 0.01) and (np.mean(data) - 1) / np.std(data, ddof=1) >= 0.01:
                keep.append(pick)
                # same ranking metric as you had: mean across epochs and time
                amp.append(np.mean(data))  # mean ratio-change over (epochs,time)
                fsize.append(np.mean(data) / np.std(data, ddof=1))

        if len(keep) == 0:
            continue
        # Plot the data:
        for pick in keep:
            onset_epoch = epochs.copy().pick(pick)[good_idx_by_ch[pick]]["stimulus onset/Relevant non-target/1500ms"]
            offset_epoch = epochs.copy().pick(pick)[good_idx_by_ch[pick]]["stimulus offset/Relevant non-target/1500ms"]
            amps = np.mean(np.squeeze(offset_epoch.get_data(tmin=0.05, tmax=0.5)), axis=1)
            plot_ieeg_image(offset_epoch.crop(-0.2, 1), pick, show=False, 
                units="HGP (norm.)", scalings=1, cmap="RdYlBu_r",                 center=1, ylim_prctile=99, logistic_cmap=True, ci=0.95)
            plt.savefig(Path(save_dir, f'sub-{sub}_ch-{pick}-offset.png'))
            plot_ieeg_image(onset_epoch.crop(-0.2, 1), pick, show=False, 
                units="HGP (norm.)", scalings=1, cmap="RdYlBu_r",
                center=1, ylim_prctile=99, logistic_cmap=True, ci=0.95)
            plt.savefig(Path(save_dir, f'sub-{sub}_ch-{pick}-onset.png'))
        
        # Test whether the offset long trials are split into ignited and non ignited trials
        event_name = "stimulus offset/Relevant non-target/1500ms"
        tmin, tmax = 0.2, 0.5

        for i, pick in enumerate(keep):
            # y: one value per trial for this electrode
            data = epochs.copy()[good_idx_by_ch[pick]][event_name].get_data(picks=pick, tmin=tmin, tmax=tmax)  # (n_trials, 1, n_times)
            y = data[:, 0, :].mean(axis=1)  # (n_trials,)

            # Fit models
            idata0 = mixture_model(y, 1, draws=1000, tune=1000, chains=2)
            idata1 = mixture_model(y, 2, draws=1000, tune=1000, chains=2)

            # Compare
            loo0, loo1, cmp, wins, delta_elpd, delta_se, pareto_k = compare_models_loo(idata0, idata1)
            winning_electrodes.append({
                    "ch": pick,
                    "delta_elpd": delta_elpd,
                    "delta_se": delta_se,#
                    "delta_elpd_norm": delta_elpd/delta_se,
                    "pareto_k": pareto_k,   # per-trial prob(comp1)
                    "fsize": fsize[i],
                    "amp": amp[i]
                })
            if wins:
                # After mixture wins:
                plot_mixture_fit(
                    idata1, y,
                    title=f"sub-{sub} {pick} | mixture vs single: WIN (ΔELPD={delta_elpd:.2f}±{delta_se:.2f})",
                    do_zscore=True,
                    bins=40,
                    n_posterior_draws=200,
                    seed=0,
                )
                plt.savefig(Path(save_dir, f"sub-{sub}_ch-{pick}_mixture_fit.png"))
                plt.close()
            else:
                plot_mixture_fit(
                    idata0, y,
                    title=f"sub-{sub} {pick} | mixture vs single: WIN (ΔELPD={delta_elpd:.2f}±{delta_se:.2f})",
                    do_zscore=True,
                    bins=40,
                    n_posterior_draws=200,
                    seed=0,
                )
                plt.savefig(Path(save_dir, f"sub-{sub}_ch-{pick}_mixture_fit.png"))
                plt.close()
    winning_electrodes = pd.DataFrame(winning_electrodes)
    winning_electrodes.to_csv(Path(save_dir, 'results.csv'))
    return None

if __name__ == "__main__":
    bids_root = "./data/bids"
    offset_analysis(ev.subjects_lists_ecog["dur"], bids_root,
                    analysis_name="offset_analysis")
