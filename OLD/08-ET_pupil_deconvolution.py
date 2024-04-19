import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from helper_function.helper_general import (baseline_scaling, max_percentage_index, equate_epochs_events,
                                            reject_bad_epochs, format_drop_logs)
from helper_function.helper_plotter import plot_ts_ci
import environment_variables as ev
import pandas as pd
from scipy.optimize import nnls


def purf_fun(times, tmax, n=10.1):
    output = (times ** n) * np.exp(-n * times / tmax)
    output[times < 0] = 0
    return output / np.max(output)


# Set the font size:
plt.rcParams.update({'font.size': 14})
from scipy.stats import ttest_1samp


def pupil_latency(parameters_file, subjects, session="1", task="prp", analysis_name="pupil_latency",
                  reject_bad_trials=True):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Load all subjects data:
    subjects_epochs = {sub: None for sub in subjects}
    # Create the directory to save the results in:
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # The soas differ between the experiments, therefore, extracting them directly from the epochs objects:
    soas = ["0", "0.116", "0.232", "0.466"]
    evoked = {soa: [] for soa in soas}
    evoked_aud = {soa: [] for soa in soas}
    evoked_vis = {soa: [] for soa in soas}
    evoked_aud_vis = {soa: [] for soa in soas}
    evoked_diff = {soa: [] for soa in soas}
    betas = {soa: [] for soa in soas}
    # Loop through each subject:
    for sub in subjects:
        print("Loading sub-{}".format(sub))
        root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                    param["data_type"])
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, task,
                                                                   param["data_type"])
        epochs = mne.read_epochs(Path(root, file_name))

        file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, "auditory",
                                                                   param["data_type"])
        aud_epochs = mne.read_epochs(Path(root, file_name))

        file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, "visual",
                                                                   param["data_type"])
        vis_epochs = mne.read_epochs(Path(root, file_name))

        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        aud_epochs.crop(param["crop"][0], epochs.times[-1])
        vis_epochs.crop(param["crop"][0], param["crop"][1])
        # Reject bad epochs according to predefined criterion:
        if reject_bad_trials:
            reject_bad_epochs(epochs,
                              baseline_window=param["baseline_window"],
                              z_thresh=param["baseline_zthresh"],
                              eyes=param["eyes"],
                              exlude_beh=param["exlude_beh"],
                              remove_blinks=param["remove_blinks"],
                              blinks_window=param["blinks_window"])

        # Extract the relevant channels:
        epochs.pick(param["picks"])
        aud_epochs.pick(param["picks"])
        vis_epochs.pick(param["picks"])

        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])
        baseline_scaling(vis_epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Create the design matrix:
        purf = purf_fun(epochs.times, 0.93, n=10.1)
        # Create the design matrix:
        sticks_times = np.arange(-0.1, 2.0, 0.1)
        # Convert those to sticks:
        sticks_matrix = np.zeros([epochs.times.shape[0], sticks_times.shape[0]])
        for ii, stick_time in enumerate(sticks_times):
            # Find the corresponding time index:
            ind = np.argmin(np.abs(epochs.times - stick_time))
            sticks_matrix[ind, ii] = 1
        # Convolve with the PURF:
        design_matrix = np.array([np.convolve(sticks_matrix[:, i], purf, mode='full')[0:sticks_matrix.shape[0]]
                                  for i in range(sticks_matrix.shape[1])]).T
        design_matrix = np.hstack((design_matrix, np.ones((design_matrix.shape[0], 1))))

        # Plot the evoked pupil responses:
        for i, soa in enumerate(soas):
            # Compute the evoked response:
            evoked[soa].append(np.mean(np.mean(epochs["/".join(["onset", soa])].get_data(copy=False), axis=1),
                                       axis=0))
            # Compute the evoked visual response:
            evoked_vis[soa].append(np.mean(np.mean(vis_epochs.get_data(copy=False), axis=1), axis=0))
            # Compute the evoked visual response:
            aud_evoked = np.zeros(epochs.times.shape[0])
            aud_epo = []
            for lock in ["onset", "offset"]:
                for soa2 in soas:
                    if lock == "onset":
                        try:
                            audio_epo = aud_epochs.copy()["/".join([lock, soa2])].crop(float(soa2) - 0.2,
                                                                                       float(
                                                                                           soa2) + 2.0)
                            baseline_scaling(audio_epo, correction_method=param["baseline"],
                                             baseline=[float(soa2) - 0.2, float(soa2)])
                            aud_epo.append(np.mean(np.mean(audio_epo.get_data(copy=False), axis=1),
                                                   axis=0))
                        except:
                            continue
            # Comput the negative least square.
            coefficients, residuals = nnls(design_matrix, evoked[soa][-1])
            betas[soa].append(coefficients[0:-1])
            soa_ind = np.argmin(np.abs(epochs.times - float(soa)))
            aud_evk = np.mean(np.array(aud_epo), axis=0)
            aud_evoked[soa_ind:soa_ind + aud_evk.shape[-1]] = aud_evk
            evoked_aud[soa].append(aud_evoked)
            evoked_aud_vis[soa].append(aud_evoked + evoked_vis[soa][-1])
            evoked_diff[soa].append(evoked[soa][-1] - evoked_aud_vis[soa][-1])

    # Crop the visual epochs
    fig, ax = plt.subplots(4, 1)
    fig_diff, ax_diff = plt.subplots(4, 1)
    fig_beta, ax_beta = plt.subplots(4, 1)
    for i, soa in enumerate(evoked.keys()):
        # Plot the evoked responses:
        plot_ts_ci(np.array(evoked[soa]), epochs.times, ev.colors["soa_onset_locked"][soa], plot_ci=True, ax=ax[i],
                   plot_single_subjects=False, label="PRP evoked")
        plot_ts_ci(np.array(evoked_vis[soa]), epochs.times, ev.colors["category"]["face"], plot_ci=True, ax=ax[i],
                   plot_single_subjects=False, label="Visual")
        plot_ts_ci(np.array(evoked_aud[soa]), epochs.times, ev.colors["category"]["object"], plot_ci=True, ax=ax[i],
                   plot_single_subjects=False, label="Audio")
        plot_ts_ci(np.array(evoked_aud_vis[soa]), epochs.times, ev.colors["category"]["letter"], plot_ci=True, ax=ax[i],
                   plot_single_subjects=False, label="Audio + Visual")
        # Plot the audio visual components
        plot_ts_ci(np.array(evoked_aud_vis[soa]), epochs.times, ev.colors["soa_onset_locked"][soa], plot_ci=True,
                   ax=ax_diff[i], plot_single_subjects=False, label="PRP - (Audio + Visual)")
        # Plot the betas:
        plot_ts_ci(np.array(betas[soa]), sticks_times, ev.colors["soa_onset_locked"][soa], plot_ci=True,
                   ax=ax_beta[i], plot_single_subjects=False, label="Deconvolution betas")

    ax[i].legend()
    ax[i].set_xlabel("Time (sec.)")
    ax[i].set_ylabel("Pupil size")
    ax_diff[i].legend()
    ax_diff[i].set_xlabel("Time (sec.)")
    ax_diff[i].set_ylabel("Pupil size")
    ax_diff[i].legend()
    ax_diff[i].set_xlabel("Time (sec.)")
    ax_diff[i].set_ylabel("Betas")
    plt.show()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\05-ET_pupil_latency_parameters.json")
    # ==================================================================================
    # PRP analysis:
    task = "prp"
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session="1",
                  analysis_name="pupil_latency", reject_bad_trials=True)
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session="1",
                  analysis_name="pupil_latency_no_rej", reject_bad_trials=False)
