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
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib.gridspec as gridspec
import pickle
from helper_function.helper_general import get_cmap_rgb_values, extract_first_bout
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
views = ['lateral', 'medial', 'rostral', 'caudal', 'ventral', 'dorsal']

# Directory of the results:
save_dir = Path(ev.bids_root, "derivatives", "decoding", "Dur", "all-dur")

# Prepare a dict for the results of each ROI:
roi_results = {}

# Loop through all pickle files:
for file in [path for path in listdir(save_dir) if path.endswith('pkl')]:
    if "all_roi" in file:
        continue
    with open(Path(save_dir, file), "rb") as fp:
        res = pickle.load(fp)
    roi_name = file.split("results-")[1].split(".pkl")[0]
    # Compute the difference between both tasks conditions:
    decoding_diff = np.mean(res["scores_tr"], axis=0) - np.mean(res["scores_ti"], axis=0)

    # Create the null distribution:
    diff_null = res["scores_shuffle_tr"] - res["scores_shuffle_ti"]

    # Compute pvalues of the difference:
    pvals = _pval_from_histogram(decoding_diff, diff_null, 1)

    # Extract the bout of significance:
    onset, offset = extract_first_bout(res['times'], pvals, 0.05, 0.05)

    if onset is not None:
        duration = offset - onset
    else:
        duration = 0

    # Plot the time series:
    fig, ax = plt.subplots(figsize=[4, 3])
    plot_decoding_results(res['times'], res["scores_tr"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["non-target"], ax=ax,
                          label="Relevant", ylim=[0.35, 1.0], onset=onset, offset=offset)
    plot_decoding_results(res['times'], res["scores_ti"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["irrelevant"], ax=ax,
                          label="Irrelevant", ylim=[0.35, 1.0], onset=None, offset=None)
    ax.axhline(0.05, res['times'][0], res['times'][-1])
    ax.set_xlim([res['times'][0], res['times'][-1]])
    ax.text(0.15, 0.9, "N={}".format(res["n_channels"]),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel("Time (sec.)")
    ax.set_ylabel("Accuracy")
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
brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_onset, cmap="jet")
for view in views:
    brain.show_view(view)
    brain.save_image(Path(save_dir, "{}_{}.png".format("onset", view)))
brain.close()

# Plot the offset of each roi on brain:
rois_offset = {roi_name: sig_rois[roi_name]["offset"] for roi_name in sig_rois.keys()}
brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_offset, cmap="jet")
for view in views:
    brain.show_view(view)
    brain.save_image(Path(save_dir, "{}_{}.png".format("offset", view)))
brain.close()

# Plot the duration of each roi on brain:
rois_duration = {roi_name: sig_rois[roi_name]["duration"] for roi_name in sig_rois.keys()}
brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_duration, cmap="jet")
for view in views:
    brain.show_view(view)
    brain.save_image(Path(save_dir, "{}_{}.png".format("duration", view)))
brain.close()
