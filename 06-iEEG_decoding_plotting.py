import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import environment_variables as ev
from mne.stats import bootstrap_confidence_interval
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib.gridspec as gridspec
import pickle
from helper_function.helper_general import get_cmap_rgb_values
from helper_function.helper_plotter import plot_decoding_results
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

save_dir = Path(ev.bids_root, "derivatives", "decoding", "Dur")
# Read the decoding data:
with open(Path(save_dir, 'all-dur', "results-all_roi.pkl"), 'rb') as f:
    roi_results = pickle.load(f)
# Read the decoding data:
with open(Path(save_dir, 'short', "results-all_roi.pkl"), 'rb') as f:
    roi_results_short = pickle.load(f)
# Read the decoding data:
with open(Path(save_dir, 'int', "results-all_roi.pkl"), 'rb') as f:
    roi_results_int = pickle.load(f)
# Read the decoding data:
with open(Path(save_dir, 'long', "results-all_roi.pkl"), 'rb') as f:
    roi_results_long = pickle.load(f)

latencies_table = pd.concat(
    [pd.DataFrame({
        "roi": roi,
        "onset": roi_results[roi]["onset"],
        "offset": roi_results[roi]["offset"],
        "duration": roi_results[roi]["duration"],
    }, index=[0])
        for roi in roi_results.keys()]
).reset_index(drop=True)
latencies_table = latencies_table[(~latencies_table["onset"].isna()) & (latencies_table["onset"] < 1.5)]
# Plot a colorbar:
norm = mpl.colors.Normalize(vmin=np.min(latencies_table["duration"].to_numpy()),
                            vmax=np.max(latencies_table["duration"].to_numpy()))
fig, ax = plt.subplots(figsize=(1, 6))
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
             cax=ax, orientation='vertical', label='Duration (s)')
fig.savefig(Path(save_dir, "colorbar.svg"),
            transparent=False, dpi=300)

# ====================================================================================================
# Plot the brain:
# ===============
# Read the annotations
annot = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', subjects_dir=ev.fs_directory)
# Onset:
# Get colors for each label:
onset_colors = get_cmap_rgb_values(latencies_table["onset"].to_numpy(), cmap="jet", center=None)
# Plot the brain:
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', hemi='lh', surf='inflated', subjects_dir=ev.fs_directory, size=(800, 600))
# Loop through each label:
for roi_i, roi in enumerate(latencies_table["roi"].to_list()):
    # Find the corresponding label:
    lbl = [l for l in annot if l.name == roi + "-lh"]
    brain.add_label(lbl[0], color=onset_colors[roi_i], borders=False)
    brain.add_label(lbl[0], color=[0, 0, 0], borders=True)
for view in views:
    brain.show_view(view)
    brain.save_image(Path(save_dir, "{}_{}.png".format("onset", view)))
brain.close()

# ===============
# offset:
# Get colors for each label:
offset_colors = get_cmap_rgb_values(latencies_table["offset"].to_numpy(), cmap="jet", center=None)
# Plot the brain:
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', hemi='lh', surf='inflated',
              subjects_dir=ev.fs_directory, size=(800, 600))
# Loop through each label:
for roi_i, roi in enumerate(latencies_table["roi"].to_list()):
    # Find the corresponding label:
    lbl = [l for l in annot if l.name == roi + "-lh"]
    brain.add_label(lbl[0], color=offset_colors[roi_i], borders=False)
    brain.add_label(lbl[0], color=[0, 0, 0], borders=True)
for view in views:
    brain.show_view(view)
    brain.save_image(Path(save_dir, "{}_{}.png".format("offset", view)))
brain.close()

# ===============
# Duration:
# Get colors for each label:
offset_colors = get_cmap_rgb_values(latencies_table["duration"].to_numpy(), cmap="jet", center=None)
# Plot the brain:
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', hemi='lh', surf='inflated',
              subjects_dir=ev.fs_directory, size=(800, 600))
# Loop through each label:
for roi_i, roi in enumerate(latencies_table["roi"].to_list()):
    # Find the corresponding label:
    lbl = [l for l in annot if l.name == roi + "-lh"]
    brain.add_label(lbl[0], color=offset_colors[roi_i], borders=False)
    brain.add_label(lbl[0], color=[0, 0, 0], borders=True)
for view in views:
    brain.show_view(view, distance="auto")
    brain.save_image(Path(save_dir, "{}_{}.png".format("duration", view)))
brain.close()

# ====================================================================================================
# Plot time series:
# ===============
for roi in roi_results.keys():
    fig_dir = Path(save_dir, "no_diff")
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    # Compute the times vector:
    times = np.linspace(-0.1, 1.5, roi_results[roi]["scores_tr"].shape[-1])

    # Create the figure:
    fig, ax = plt.subplots(figsize=[8.3, 8.3])
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0)
    # Create each subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    # Remove the underlying axes:
    ax.remove()

    # All duration decoding accuracy:
    plot_decoding_results(times, roi_results[roi]["scores_targets"], ci=0.95, smooth_ms=50,
                          color=[0, 0, 1], ax=ax1, label=None, ylim=[0.35, 1.0],
                          onset=roi_results[roi]["onset"], offset=roi_results[roi]["offset"])

    plot_decoding_results(times, roi_results[roi]["scores_tr"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["non-target"], ax=ax1, label=None, ylim=[0.35, 1.0],
                          onset=roi_results[roi]["onset"], offset=roi_results[roi]["offset"])

    plot_decoding_results(times, roi_results[roi]["scores_ti"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["irrelevant"], ax=ax1, label=None, ylim=[0.35, 1.0],
                          onset=None, offset=None)
    # Short durations:
    plot_decoding_results(times, roi_results_short[roi]["scores_tr"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["non-target"], ax=ax2, label=None, ylim=[0.35, 1.0],
                          onset=roi_results_short[roi]["onset"], offset=roi_results_short[roi]["offset"])
    plot_decoding_results(times, roi_results_short[roi]["scores_ti"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["irrelevant"], ax=ax2, label=None, ylim=[0.35, 1.0],
                          onset=None, offset=None)
    # Add a box to show the stimulus duration:
    pres_time = np.arange(0, 0.5, times[1] - times[0])
    ax2.fill_between(pres_time, 0.35, 1.0,
                     color=[0.5, 0.5, 0.5], alpha=.2)
    # Intermediate durations:
    plot_decoding_results(times, roi_results_int[roi]["scores_tr"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["non-target"], ax=ax3, label=None, ylim=[0.35, 1.0],
                          onset=roi_results_int[roi]["onset"], offset=roi_results_int[roi]["offset"])
    plot_decoding_results(times, roi_results_int[roi]["scores_ti"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["irrelevant"], ax=ax3, label=None, ylim=[0.35, 1.0],
                          onset=None, offset=None)
    # Add a box to show the stimulus duration:
    pres_time = np.arange(0, 1.0, times[1] - times[0])
    ax3.fill_between(pres_time, 0.35, 1.0,
                     color=[0.5, 0.5, 0.5], alpha=.2)
    # Long durations:
    plot_decoding_results(times, roi_results_long[roi]["scores_tr"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["non-target"], ax=ax4, label=None, ylim=[0.35, 1.0],
                          onset=roi_results_long[roi]["onset"], offset=roi_results_long[roi]["offset"])
    plot_decoding_results(times, roi_results_long[roi]["scores_ti"], ci=0.95, smooth_ms=50,
                          color=ev.colors["task_relevance"]["irrelevant"], ax=ax4, label=None, ylim=[0.35, 1.0],
                          onset=None, offset=None)
    # Add a box to show the stimulus duration:
    pres_time = np.arange(0, 1.5, times[1] - times[0])
    ax4.fill_between(pres_time, 0.35, 1.0,
                     color=[0.5, 0.5, 0.5], alpha=.2)

    ax1.text(0.8, 0.9, "N={}".format(roi_results[roi]["n_channels"]),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    # Hide x-axis labels and ticks for the first three subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
    ax1.set_xlim([times[0], times[-1]])
    ax1.set_ylim([0.4, 1.0])
    ax4.set_xlabel("Time (sec.)")
    ax4.set_ylabel("Accuracy")
    plt.tight_layout()
    if roi_results[roi]["onset"] is not None:
        fig.savefig(Path(save_dir, "{}_decoding.svg".format(roi)),
                    transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "{}_decoding.png".format(roi)),
                    transparent=True, dpi=300)
    else:
        fig.savefig(Path(fig_dir, "{}_decoding.svg".format(roi)),
                    transparent=True, dpi=300)
        fig.savefig(Path(fig_dir, "{}_decoding.png".format(roi)),
                    transparent=True, dpi=300)

    plt.close()
