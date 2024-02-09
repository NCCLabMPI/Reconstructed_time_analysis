import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helper_function.helper_general import generate_gaze_map, deg_to_pix
import pandas as pd
import os
import environment_variables as ev
from helper_function.helper_plotter import soa_boxplot
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Set the font size:
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'font.size': 14})
dpi = 300
figure_height = 8.3


def check_plots(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Prepare a dataframe to store the fixation proportion:
    check_values = pd.DataFrame()
    # Prepare a list to store the fixation heatmaps:
    fixation_heatmaps = []
    # Loop through each subject:
    for sub in subjects:
        print(sub)
        # ===========================================
        # Data loading:
        # Load the epochs:
        root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + param["session"],
                    param["data_type"])
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, param["session"], param["task"],
                                                                   param["data_type"])
        epochs = mne.read_epochs(Path(root, file_name))
        # Decimate
        epochs.decimate(int(epochs.info["sfreq"] / param["decim_freq"]))

        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])

        # Compute the gaze map for this subject:
        fixation_heatmaps.append(generate_gaze_map(epochs, 1080, 1920, sigma=20))
        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract data locked to the visual stimulus:
                        epochs_cropped = epochs.copy().crop(param["crop"][0], param["crop"][1])

                        # Extract the data and average across left and right eye:
                        fixation_data = np.mean(
                            epochs_cropped.copy()["/".join([task_rel, duration, lock, soa])].pick(["fixdist_left",
                                                                                                   "fixdist_right"]).get_data(
                                copy=False),
                            axis=1)
                        blinks_data = np.mean(
                            epochs_cropped.copy()["/".join([task_rel, duration, lock, soa])].pick(["blink_left",
                                                                                                   "blink_right"]).get_data(
                                copy=False),
                            axis=1)
                        # Compute the fixation proportion:
                        fix_prop = np.mean(np.mean(fixation_data < param["fixdist_thresh_deg"], axis=1))
                        # Compute the blink rate:
                        blink_rate = np.mean(np.sum(np.diff(blinks_data, axis=1) == 1, axis=1))

                        # Add to data frame using pd.concat:
                        check_values = pd.concat([check_values,
                                                  pd.DataFrame({"sub_id": sub,
                                                                "task_relevance": task_rel,
                                                                "duration": float(duration),
                                                                "SOA_lock": lock,
                                                                "soa": float(soa),
                                                                "onset_SOA": float(soa) + float(duration)
                                                                if lock == "offset" else float(soa),
                                                                "fixation_proportion": fix_prop,
                                                                "blink_rate": blink_rate},
                                                               index=[0])])
    check_values = check_values.reset_index(drop=True)
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "check_plots", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    check_values.to_csv(Path(save_dir, "check_values.csv"))

    # =========================================================================
    # Plot the results:
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "check_plots", "figures")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # =========================================================================
    # Fixation proportion:
    # Across task relevances:
    fig_all, ax_all = soa_boxplot(check_values,
                                  "fixation_proportion",
                                  fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                            param["screen_res"][0]])
    # Task relevant:
    fig_tr, ax_tr = soa_boxplot(check_values[check_values["task_relevance"] == 'non-target'],
                                "fixation_proportion",
                                fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                          param["screen_res"][0]])
    # Task irrelevant:
    fig_ti, ax_ti = soa_boxplot(check_values[check_values["task_relevance"] == 'irrelevant'],
                                "fixation_proportion",
                                fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                          param["screen_res"][0]])
    # Set the y limit to be the same for both plots:
    lims = [[ax_all[0].get_ylim()[0], ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]],
            [ax_all[0].get_ylim()[1], ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
    max_lims = [min(min(lims)), max(max(lims))]
    ax_all[0].set_ylim(max_lims)
    ax_tr[0].set_ylim(max_lims)
    ax_ti[0].set_ylim(max_lims)
    # Axes decoration:
    fig_all.suptitle("Fixation proportion per SOA condition")
    fig_tr.suptitle("Relevant non-target")
    fig_ti.suptitle("Irrelevant non-target")
    fig_all.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_tr.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_ti.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_all.text(0, 0.5, 'Fixation proportion', ha='center', va='center', fontsize=18, rotation=90)
    fig_tr.text(0, 0.5, 'Fixation proportion', ha='center', va='center', fontsize=18, rotation=90)
    fig_ti.text(0, 0.5, 'Fixation proportion', ha='center', va='center', fontsize=18, rotation=90)
    fig_all.savefig(Path(save_dir, "fixation_all.svg"), transparent=True, dpi=dpi)
    fig_all.savefig(Path(save_dir, "fixation_all.png"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "fixation_tr.svg"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "fixation_tr.png"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti.svg"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti.png"), transparent=True, dpi=dpi)
    plt.close(fig_all)
    plt.close(fig_tr)
    plt.close(fig_ti)

    # =========================================================================
    # Blink rate:
    # Across task relevances:
    fig_all, ax_all = soa_boxplot(check_values,
                                  "blink_rate",
                                  fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                            param["screen_res"][0]])
    # Task relevant:
    fig_tr, ax_tr = soa_boxplot(check_values[check_values["task_relevance"] == 'non-target'],
                                "blink_rate",
                                fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                          param["screen_res"][0]])
    # Task irrelevant:
    fig_ti, ax_ti = soa_boxplot(check_values[check_values["task_relevance"] == 'irrelevant'],
                                "blink_rate",
                                fig_size=[figure_height, figure_height * param["screen_res"][1] /
                                          param["screen_res"][0]])
    # Set the y limit to be the same for both plots:
    lims = [[ax_all[0].get_ylim()[0], ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]],
            [ax_all[0].get_ylim()[1], ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
    max_lims = [min(min(lims)), max(max(lims))]
    ax_all[0].set_ylim(max_lims)
    ax_tr[0].set_ylim(max_lims)
    ax_ti[0].set_ylim(max_lims)
    # Axes decoration:
    fig_all.suptitle("Blink rate per SOA condition")
    fig_tr.suptitle("Relevant non-target")
    fig_ti.suptitle("Irrelevant non-target")
    fig_all.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_tr.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_ti.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_all.text(0, 0.5, 'Blink rate (Hz)', ha='center', va='center', fontsize=18, rotation=90)
    fig_tr.text(0, 0.5, 'Blink rate (Hz)', ha='center', va='center', fontsize=18, rotation=90)
    fig_ti.text(0, 0.5, 'Blink rate (Hz)', ha='center', va='center', fontsize=18, rotation=90)
    fig_all.savefig(Path(save_dir, "blink_rate_all.svg"), transparent=True, dpi=dpi)
    fig_all.savefig(Path(save_dir, "blink_rate_all.png"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "blink_rate_tr.svg"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "blink_rate_tr.png"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "blink_rate_ti.svg"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "blink_rate_ti.png"), transparent=True, dpi=dpi)
    plt.close(fig_all)
    plt.close(fig_tr)
    plt.close(fig_ti)

    # Plot the dwell time image:
    hists = np.nanmean(np.array(fixation_heatmaps), axis=0)
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[figure_height,
                                                                                   figure_height *
                                                                                   param["screen_res"][1] /
                                                                                   param["screen_res"][0]])
    vmin = np.nanpercentile(hists, 5)
    vmax = np.nanmax(hists)
    extent = [0, param["screen_res"][0], param["screen_res"][1], 0]  # origin is the top left of the screen
    # Plot heatmap
    cmap = plt.get_cmap("RdYlBu_r")
    im = ax3.imshow(
        hists,
        aspect="equal",
        cmap=cmap,
        alpha=1,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    # Calculate the sizes in pixels:
    center = [param["screen_res"][0] / 2, param["screen_res"][1] / 2]
    fixation_radius = deg_to_pix(param["fixdist_thresh_deg"], param["screen_distance_cm"],
                                 param["screen_size_cm"], param["screen_res"])
    stim_size = deg_to_pix(param["stim_size_deg"], param["screen_distance_cm"],
                           param["screen_size_cm"], param["screen_res"])
    stim_img = mpimg.imread('FACE01.png')
    stim_extent = [center[0] - stim_size / 2, center[0] + stim_size / 2,
                   center[1] - stim_size / 2, center[1] + stim_size / 2]
    ax3.imshow(stim_img, extent=stim_extent, alpha=0.5)
    circle = patches.Circle(center, fixation_radius, edgecolor='red', facecolor='none', linewidth=2)
    # Add the circle to the plot
    ax3.add_patch(circle)
    ax3.set_title("Gaze heatmap")
    ax3.set_xlabel("X position (pix.)")
    ax3.set_ylabel("Y position (pix.)")
    fig3.colorbar(im, ax=ax3, shrink=0.8, label="Dwell time (seconds)")
    ax3.set_xlim(0, param["screen_res"][0])
    ax3.set_ylim(0, param["screen_res"][1])
    fig3.savefig(Path(save_dir, "fixation_map.svg"), transparent=True, dpi=dpi)
    fig3.savefig(Path(save_dir, "fixation_map.png"), transparent=True, dpi=dpi)
    plt.close(fig3)

    return None


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121", "SX123"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\02-check_plots_parameters.json ")
    check_plots(parameters, subjects_list)
