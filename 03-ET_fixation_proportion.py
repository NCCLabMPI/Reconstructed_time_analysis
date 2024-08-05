import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helper_function.helper_general import generate_gaze_map, deg_to_pix, equate_epochs_events
import pandas as pd
import os
import environment_variables as ev
from helper_function.helper_plotter import soa_boxplot
import matplotlib.image as mpimg
import matplotlib.patches as patches
from scipy.ndimage import uniform_filter1d

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


def check_plots(parameters_file, subjects, session="1", task="prp"):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Prepare a dataframe to store the fixation proportion:
    check_values = pd.DataFrame()
    blinks_bfreafter = pd.DataFrame()
    # Prepare a list to store the fixation heatmaps:
    fixation_heatmaps = []
    fixation_proportion = pd.DataFrame()
    # Loop through each subject:
    for sub in subjects:
        print("Loading sub-{}".format(sub))
        if isinstance(session, list):
            epochs = []
            for ses in session:
                root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + ses,
                            param["data_type"])
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, ses, task,
                                                                           param["data_type"])
                epochs.append(mne.read_epochs(Path(root, file_name)))
            # Equate the epochs events.
            epochs = equate_epochs_events(epochs)
            epochs = mne.concatenate_epochs(epochs, add_offset=True)
        else:
            root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                        param["data_type"])
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, task,
                                                                       param["data_type"])
            epochs = mne.read_epochs(Path(root, file_name))
        # Decimate
        if param["decim_freq"] is not None:
            epochs.decimate(int(epochs.info["sfreq"] / param["decim_freq"]))

        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])

        # Compute the gaze map for this subject:
        fixation_heatmaps.append(generate_gaze_map(epochs, 1080, 1920, sigma=20))

        # Loop through each of the duration conditions:
        for dur in param["durations"]:
            # Crop the data:
            epo = epochs.copy()[dur].crop(0, float(dur))
            # Extract the fixation mask:
            fixation_mask = np.any(epo.copy().pick(["fixation_left", "fixation_right"]).get_data(copy=False), axis=1)
            # Average fixation distance across both eyes:
            fixation_data = np.mean(epo.pick(["fixdist_left", "fixdist_right"]).get_data(copy=False), axis=1)
            # Extract the samples that are less than the defined threshold:
            less_than_thresh = np.array(fixation_data < param["fixdist_thresh_deg"]).astype(float)
            # Keep only the fixations:
            less_than_thresh[~fixation_mask] = np.nan
            # Compute the fixation proportion across time and trials:
            fix_prop = np.nanmean(np.nanmean(less_than_thresh, axis=1))
            # Add to the table:
            fixation_proportion = pd.concat([fixation_proportion, pd.DataFrame({
                "sub_id": sub,
                "duration": dur,
                "fixation_proportion": fix_prop
            }, index=[0])])

        # Compute the fixation proportion across durations betweenn 0 and 2 secs:
        epo = epochs.copy().crop(0, 2.0)
        # Extract the fixation mask:
        fixation_mask = np.any(epo.copy().pick(["fixation_left", "fixation_right"]).get_data(copy=False), axis=1)
        # Average fixation distance across both eyes:
        fixation_data = np.mean(epo.pick(["fixdist_left", "fixdist_right"]).get_data(copy=False), axis=1)
        # Extract the samples that are less than the defined threshold:
        less_than_thresh = np.array(fixation_data < param["fixdist_thresh_deg"]).astype(float)
        # Keep only the fixations:
        less_than_thresh[~fixation_mask] = np.nan
        # Compute the fixation proportion across time and trials:
        fix_prop = np.nanmean(np.nanmean(less_than_thresh, axis=1))
        # Add to the table:
        fixation_proportion = pd.concat([fixation_proportion, pd.DataFrame({
            "sub_id": sub,
            "duration": "all",
            "fixation_proportion": fix_prop
        }, index=[0])])

    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "fixation_proportion", task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    fixation_proportion.to_csv(Path(save_dir, "fixation_proportion.csv"))

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
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\03-ET_fixation_proportion_parameters.json")
    # ==================================================================================
    # PRP analysis:
    task = "prp"
    check_plots(parameters, ev.subjects_lists_et[task], task="prp", session="1")
    # ==================================================================================
    # Introspection analysis:
    task = "introspection"
    check_plots(parameters, ev.subjects_lists_et[task], task=task, session=["2", "3"])
