import mne
import os
from os import listdir
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import environment_variables as ev
from mne.stats import bootstrap_confidence_interval
from mne.stats.cluster_level import _pval_from_histogram
import pickle
from helper_function.helper_general import extract_first_bout, cluster_test
from helper_function.helper_plotter import plot_decoding_results, plot_rois
import pandas as pd

# Set the font size:
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Set list of views:
views = {'side': {"azimuth": 180, "elevation": 90}, 'front': {"azimuth": 130, "elevation": 90},
         "ventral": {"azimuth": 90, "elevation": 180}}

subfolders = ["decoding_no_pseudo", "decoding_no_pseudo_5ms", "decoding_no_pseudo_5ms_acc",
              "decoding_no_pseudo_acc", "decoding_pseudotrials", "decoding_pseudotrials_5ms_acc",
              "decoding_pseudotrials_acc"]

for fl in subfolders:
    # Directory of the results:
    save_dir = Path(ev.bids_root, "derivatives", "decoding_no_uw", "Dur", fl)

    # Prepare a dict for the results of each ROI:
    roi_results = {}

    # Loop through all pickle files:
    for file in [path for path in listdir(save_dir) if path.endswith('pkl')]:
        if "all_roi" in file:
            continue
        with open(Path(save_dir, file), "rb") as fp:
            res = pickle.load(fp)
        roi_name = file.split("results-")[1].split(".pkl")[0]
        if res["n_channels"] < 10:
            continue
        # Compute the difference between both tasks conditions:
        decoding_diff = np.mean(res["scores_tr"], axis=0) - np.mean(res["scores_ti"], axis=0)

        # Create the null distribution:
        diff_null = res["scores_shuffle_tr"] - res["scores_shuffle_ti"]

        # Compute pvalues of the difference:
        x_zscored, h0_zscore, clusters, cluster_pv, p_values, h0 = cluster_test(decoding_diff, diff_null,
                                                                                z_threshold=1.5,
                                                                                do_zscore=True)
        if any(p_values < 0.05):
            msk = np.array(p_values < 0.05, dtype=int)
            onset = res["times"][np.where(np.diff(msk) == 1)[0]]
            offset = res["times"][np.where(np.diff(msk) == -1)[0][0]]
            duration = offset - onset
        else:
            onset = None
            offset = None
            duration = 0

        pvals = _pval_from_histogram(decoding_diff, diff_null, 1)

        # Extract the bout of significance:
        onset, offset = extract_first_bout(res['times'], pvals, 0.05, 0.04)

        if onset is not None:
            duration = offset - onset
        else:
            duration = 0

        #  Plot the time series:
        fig, ax = plt.subplots(figsize=[4, 3])
        plot_decoding_results(res['times'], res["scores_tr"], ci=0.95, smooth_ms=20,
                              color=ev.colors["task_relevance"]["non-target"], ax=ax,
                              label="Relevant", ylim=[0.35, 1.0], onset=onset, offset=offset)
        plot_decoding_results(res['times'], res["scores_ti"], ci=0.95, smooth_ms=20,
                              color=ev.colors["task_relevance"]["irrelevant"], ax=ax,
                              label="Irrelevant", ylim=[0.35, 1.0], onset=None, offset=None)
        ax.axhline(0.05, res['times'][0], res['times'][-1])
        ax.set_xlim([res['times'][0], res['times'][-1]])
        ax.text(0.15, 0.9, "N={}".format(res["n_channels"]),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel("Time (sec.)")
        ax.set_ylabel("AUC")
        ax.spines[['right', 'top']].set_visible(False)
        plt.legend(frameon=False)
        plt.tight_layout()
        if onset is not None:
            fig_dir = Path(save_dir, "figures", "significant")
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
        else:
            fig_dir = Path(save_dir, "figures", "non-significant")
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding.svg"),
                    transparent=True, dpi=300)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding.png"),
                    transparent=True, dpi=300)
        plt.close()

        # Store the results here:
        roi_results[roi_name] = {"onset": onset, "offset": offset, "duration": duration}

    # Extract the significant ROIs:
    sig_rois = {roi_name: roi_results[roi_name] for roi_name in roi_results.keys()
                if roi_results[roi_name]["onset"] is not None}

    # Plot the onset of each roi on brain:
    rois_onset = {roi_name: sig_rois[roi_name]["onset"] for roi_name in sig_rois.keys()}
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_onset, cmap="Reds")
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("onset", view)))
    brain.close()

    # Plot the offset of each roi on brain:
    rois_offset = {roi_name: sig_rois[roi_name]["offset"] for roi_name in sig_rois.keys()}
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_offset, cmap="Reds")
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("offset", view)))
    brain.close()

    # Plot the duration of each roi on brain:
    rois_duration = {roi_name: sig_rois[roi_name]["duration"] for roi_name in sig_rois.keys()}
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_duration, cmap="Reds")
    for view in views.keys():
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("duration", view)))
    brain.close()
    # Plot a colorbar:
    norm = mpl.colors.Normalize(vmin=min(rois_duration.values()),
                                vmax=min(rois_duration.values()))
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Reds'),
                 cax=ax, orientation='horizontal', label='Duration (s)')
    fig.savefig(Path(save_dir, "colorbar.svg"),
                transparent=False, dpi=300)
    plt.close()
