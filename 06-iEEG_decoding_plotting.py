import os
from os import listdir
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import environment_variables as ev
import pickle
import matplotlib.colors as mcolors
from helper_function.helper_general import cluster_test
from helper_function.helper_plotter import plot_decoding_results, plot_rois, get_color_mapping
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

subfolders = ["decoding_auc", "decoding_acc"]
alpha = 0.01
smooth_ms = 40
for fl in subfolders:
    # Directory of the results:
    save_dir = Path(ev.bids_root, "derivatives", "decoding_10ms_alpha01", "Dur", fl)

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
            # Store the results here:
            roi_results[roi_name] = {
                "onset": None,
                "offset": None,
                "duration": None,
                "h0_tr": True,
                "max_tr": 0,
                "h0_ti": True,
                "max_ti": 0,
                "n_channels": res["n_channels"]
            }
            continue
        if roi_name == 'S_front_inf':
            print('A')
        # Compute the difference between both tasks conditions:
        decoding_diff = np.mean(res["scores_tr"], axis=0) - np.mean(res["scores_ti"], axis=0)

        # Create the null distribution:
        diff_null = res["scores_shuffle_tr"] - res["scores_shuffle_ti"]

        # Compute pvalues of the difference:
        x_zscored, h0_zscore, clusters, cluster_pv, p_values, h0 = cluster_test(decoding_diff, diff_null,
                                                                                z_threshold=1.5,
                                                                                do_zscore=True)
        if any(p_values < alpha):
            msk = np.array(p_values < alpha, dtype=int)
            onset = res["times"][np.where(np.diff(msk) == 1)[0] + 1]
            offset = res["times"][np.where(np.diff(msk) == -1)[0]]
            if offset.size == 0:
                offset = np.array([res["times"][-1]])
            duration = offset - onset
        else:
            onset = None
            offset = None
            duration = 0
        if onset is not None and len(onset) > 1:
            print("A")

        # pvals = _pval_from_histogram(decoding_diff, diff_null, 1)
        # onset, offset = extract_first_bout(res['times'], pvals, 0.05, 0.04)
        # if onset is not None:
        #     duration = offset - onset
        # else:
        #     duration = None

        #  Plot the time series:
        fig, ax = plt.subplots(figsize=[4, 3])
        plot_decoding_results(res['times'], res["scores_tr"], ci=0.95, smooth_ms=smooth_ms,
                              color=ev.colors["task_relevance"]["non-target"], ax=ax,
                              label="Relevant", ylim=[0.35, 1.0], onset=onset, offset=offset)
        plot_decoding_results(res['times'], res["scores_ti"], ci=0.95, smooth_ms=smooth_ms,
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

        # Plot the null distribution for reference:
        # Prepare a grid for the heatmap
        num_bins = 200
        hist, xedges, yedges = np.histogram2d(diff_null.flatten(),
                                              np.repeat(np.arange(diff_null.shape[1]), diff_null.shape[0]),
                                              bins=[num_bins, len(res['times'])],
                                              range=[[-1, 1], [0, len(res['times'])]])  #
        # Create the meshgrid for the surface plot
        xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2,
                                 yedges[:-1] + (yedges[1] - yedges[0]) / 2, indexing="ij")
        # Create a 3D figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(xpos, ypos, hist, cmap='viridis')
        ax.set_xlabel('Decoding Accuracy')
        ax.set_ylabel('Time Points')
        ax.set_zlabel('Density')
        ax.set_title('Null Distribution Over Time')
        fig.savefig(Path(fig_dir, f"{roi_name}_null_dist.png"),
                    transparent=True, dpi=300)
        plt.close()

        # Plot the significance of the decoding for the task relevant and irrelevant separately:

        # ======================================================================================
        # Task relevant
        x_zscored, h0_zscore, clusters, cluster_pv, p_values, h0 = cluster_test(np.mean(res["scores_tr"], axis=0),
                                                                                res["scores_shuffle_tr"],
                                                                                z_threshold=1.5,
                                                                                do_zscore=True)
        if any(p_values < alpha):
            msk = np.array(p_values < alpha, dtype=int)
            onset_tr = res["times"][np.where(np.diff(msk) == 1)[0] + 1]
            offset_tr = res["times"][np.where(np.diff(msk) == -1)[0]]
            if offset_tr.size == 0:
                offset_tr = np.array([res["times"][-1]])
            duration_tr = offset_tr - onset_tr
            h0_tr = False
            max_tr = np.max(np.mean(res["scores_tr"], axis=0))
        else:
            onset_tr = None
            offset_tr = None
            duration_tr = 0
            h0_tr = True
            max_tr = 0

        #  Plot the time series:
        fig, ax = plt.subplots(figsize=[4, 3])
        plot_decoding_results(res['times'], res["scores_tr"], ci=0.95, smooth_ms=smooth_ms,
                              color=ev.colors["task_relevance"]["non-target"], ax=ax,
                              label="Relevant", ylim=[0.35, 1.0], onset=onset_tr, offset=offset_tr)
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
        fig_dir = Path(save_dir, "figures", "per_task")
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding_tr.svg"),
                    transparent=True, dpi=300)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding_tr.png"),
                    transparent=True, dpi=300)
        plt.close()

        # ======================================================================================
        # Task irrelevant
        # Compute pvalues:
        x_zscored, h0_zscore, clusters, cluster_pv, p_values, h0 = cluster_test(np.mean(res["scores_ti"], axis=0),
                                                                                res["scores_shuffle_ti"],
                                                                                z_threshold=1.5,
                                                                                do_zscore=True)
        if any(p_values < alpha):
            msk = np.array(p_values < alpha, dtype=int)
            onset_ti = res["times"][np.where(np.diff(msk) == 1)[0] + 1]
            offset_ti = res["times"][np.where(np.diff(msk) == -1)[0]]
            if offset_ti.size == 0:
                offset_ti = np.array([res["times"][-1]])
            duration_ti = offset_ti - onset_ti
            h0_ti = False
            max_ti = np.max(np.mean(res["scores_tr"], axis=0))
        else:
            onset_ti = None
            offset_ti = None
            duration_ti = 0
            h0_ti = True
            max_ti = 0

        #  Plot the time series:
        fig, ax = plt.subplots(figsize=[4, 3])
        plot_decoding_results(res['times'], res["scores_ti"], ci=0.95, smooth_ms=smooth_ms,
                            color=ev.colors["task_relevance"]["irrelevant"], ax=ax,
                            label="Irrelevant", ylim=[0.35, 1.0], onset=onset_ti, offset=offset_ti)

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
        fig_dir = Path(save_dir, "figures", "per_task")
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding_ti.svg"),
                    transparent=True, dpi=300)
        fig.savefig(Path(fig_dir, f"{roi_name}_decoding_ti.png"),
                    transparent=True, dpi=300)
        plt.close()

        # Store the results here:
        roi_results[roi_name] = {
            "onset": onset[0] if isinstance(onset, np.ndarray) else onset,
            "offset": offset[0] if isinstance(offset, np.ndarray) else offset,
            "duration": duration[0] if isinstance(duration, np.ndarray) else duration,
            "h0_tr": h0_tr,
            "max_tr": max_tr,
            "h0_ti": h0_ti,
            "max_ti": max_ti,
            "n_channels": res["n_channels"]
        }

    # Prepare a dictionary to plot the ROI with too few electrodes in a different colour:
    sparse_roi_colors = {roi: [0, 0, 0] for roi in roi_results if roi_results[roi]['n_channels'] == 0}

    # Extract the significant ROIs:
    sig_rois = {roi_name: roi_results[roi_name] for roi_name in roi_results.keys()
                if roi_results[roi_name]["onset"] is not None}
    if not sig_rois:
        continue

    # Plot the onset of each roi on brain:
    rois_onset = {roi_name: sig_rois[roi_name]["onset"] for roi_name in sig_rois.keys()}
    # Convert to RGB values:
    rois_colors = get_color_mapping(rois_onset, color_map='Reds', min_prctile=0.2)
    rois_colors.update(sparse_roi_colors)
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors)
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("onset", view)))
    brain.close()

    # Plot the offset of each roi on brain:
    rois_offset = {roi_name: sig_rois[roi_name]["offset"] for roi_name in sig_rois.keys()}
    rois_colors = get_color_mapping(rois_offset, color_map='Reds', min_prctile=0.2)
    rois_colors.update(sparse_roi_colors)
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors)
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("offset", view)))
    brain.close()

    # Plot the duration of each roi on brain:
    rois_duration = {roi_name: sig_rois[roi_name]["duration"] for roi_name in sig_rois.keys()}
    rois_colors = get_color_mapping(rois_duration, color_map='Reds', min_prctile=0.2)
    rois_colors.update(sparse_roi_colors)
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors)
    for view in views.keys():
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("duration", view)))
    brain.close()
    # Plot a colorbar:
    min_val = min(rois_duration.values())
    max_val = max(rois_duration.values())
    norm = mpl.colors.Normalize(vmin=min_val - (max_val - min_val) * 0.2,
                                vmax=max_val)
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Reds'),
                 cax=ax, orientation='horizontal', label='Duration (s)')
    fig.savefig(Path(save_dir, "colorbar.svg"),
                transparent=False, dpi=300)
    plt.close()

    # ===========================================================================================
    # Plot task relevant on brain:
    # Extract the significant ROIs:
    tr_rois = {roi_name: roi_results[roi_name] for roi_name in roi_results.keys()
               if roi_results[roi_name]["h0_tr"] is False}
    if not sig_rois:
        continue

    # Plot the onset of each roi on brain:
    tr_maxs = {roi_name: tr_rois[roi_name]["max_tr"] for roi_name in tr_rois.keys()}
    rois_colors = get_color_mapping(tr_maxs, color_map="Oranges", min_prctile=0.2)
    rois_colors.update(sparse_roi_colors)
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors)
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("task_relevant", view)))
    brain.close()
    # Plot a colorbar:
    min_val = min(tr_maxs.values())
    max_val = max(tr_maxs.values())
    norm = mpl.colors.Normalize(vmin=min_val - (max_val - min_val) * 0.2,
                                vmax=max_val)
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Oranges'),
                 cax=ax, orientation='vertical', label='Duration (s)')
    fig.savefig(Path(save_dir, "colorbar_tr.svg"),
                transparent=False, dpi=300)
    plt.close()

    # ===========================================================================================
    # Plot task irrelevant on brain:
    # Extract the significant ROIs:
    ti_rois = {roi_name: roi_results[roi_name] for roi_name in roi_results.keys()
               if roi_results[roi_name]["h0_ti"] is False}
    if not sig_rois:
        continue
    # Plot the onset of each roi on brain:
    ti_maxs = {roi_name: ti_rois[roi_name]["max_ti"] for roi_name in ti_rois.keys()}
    rois_colors = get_color_mapping(ti_maxs, color_map="Greens", min_prctile=0.2)
    rois_colors.update(sparse_roi_colors)
    brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors)
    for view in views:
        brain.show_view(**views[view])
        brain.save_image(Path(save_dir, "{}_{}.png".format("task_irrelevant", view)))
    brain.close()
    # Plot a colorbar:
    min_val = min(ti_maxs.values())
    max_val = max(ti_maxs.values())
    norm = mpl.colors.Normalize(vmin=min_val - (max_val - min_val) * 0.2,
                                vmax=max_val)
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greens'),
                 cax=ax, orientation='vertical', label='Duration (s)')
    fig.savefig(Path(save_dir, "colorbar_ti.svg"),
                transparent=False, dpi=300)
    plt.close()
