import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling, max_percentage_index
from eye_tracker.plotter_functions import plot_ts_ci, plot_within_subject_boxplot
import environment_variables as ev
import pandas as pd

# Set the font size:
plt.rcParams.update({'font.size': 14})


def pupil_latency(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Load all subjects data:
    subjects_epochs = {sub: None for sub in subjects}
    # Create the directory to save the results in:
    save_dir = Path(ev.bids_root, "derivatives", "pupil_latency")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Loop through each subject:
    for sub in subjects:
        print("Loading sub-{}".format(sub))
        root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + param["session"],
                    param["data_type"])
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, param["session"], param["task"],
                                                                      param["data_type"],
                                                                      param["epoch_name"])
        epochs = mne.read_epochs(Path(root, file_name))
        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        # Extract the relevant channels:
        epochs.pick(param["picks"])
        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])
        subjects_epochs[sub] = epochs

    # ==================================================================================================================
    # SOAs comparisons:
    # ===========================================================
    # Plot the pupil response separately to each SOA for the onset locked trials:
    lock = "onset"
    fig = plt.figure(figsize=[10, 11.7 / 2], constrained_layout=True)
    spec = fig.add_gridspec(ncols=4, nrows=1)
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[0, 3])
    latencies = []
    for soa in param["soas"]:
        evks = []
        lats = []
        vals = []
        # Loop through each subject:
        for sub in subjects_epochs.keys():
            # Average the data across both eyes:
            data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock])], axis=1)
            # Remove any trials containing Nan:
            data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
            # Remove any trials containing Nan:
            data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
            evk = np.mean(data, axis=0)
            # Find the time at % peak:
            ind, val = max_percentage_index(evk, 90)
            lats.append(subjects_epochs[sub].times[ind])
            vals.append(val)
            latencies.append(pd.DataFrame({
                "sub_id": sub,
                "SOA_lock": "onset",
                "SOA": soa,
                "SOA_float": float(soa),
                "latency": subjects_epochs[sub].times[ind],
                "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                "value": val
            }, index=[0]))
            evks.append(evk)
        # Plot the evoked pupil response:
        plot_ts_ci(np.array(evks), epochs.times, ev.colors["soa_onset_locked"][soa], plot_ci=False, ax=ax1, label=soa)
        # Plot the latency:
        mean_lat = np.mean(lats)
        ax1.vlines(x=float(soa), ymin=-0.02, ymax=-0.01, linestyle="-",
                   color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
        ax1.vlines(x=mean_lat, ymin=-0.02,
                   ymax=np.mean(np.array(evks), axis=0)[np.argmin(np.abs(epochs.times - mean_lat))],
                   linestyle="--",
                   color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
    ax1.set_ylim([-0.02, 0.09])
    ax1.legend()
    ax1.set_xlabel("Time (sec.)")
    ax1.set_ylabel("Pupil size (norm.)")
    # Plot the latency as a function of SOA:
    latencies = pd.concat(latencies).reset_index(drop=True)
    plot_within_subject_boxplot(latencies, "sub_id", "SOA_float", "latency_aud",
                                positions="SOA_float", ax=ax2, cousineau_correction=False, title="", xlabel="SOA",
                                ylabel=r'$\tau_{\mathrm{audio}}$', xlim=[-0.1, 0.6], width=0.1,
                                face_colors=[ev.colors["soa_onset_locked"][soa] for soa in param["soas"]])
    ax2.set_ylim([0, 1.5])
    plt.suptitle("Pupil peak latency")
    fig.savefig(Path(save_dir, "pupil_latency_titr_{}.svg".format(lock)), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "pupil_latency_titr_{}.png".format(lock)), transparent=True, dpi=300)
    plt.close()

    # ===========================================================
    # Plot the pupil response separately to each SOA for the onset locked trials separately for each task relevance
    # conditions:
    lock = "onset"
    for task in param["task_relevance"]:
        fig = plt.figure(figsize=[10, 11.7 / 4], constrained_layout=True)
        spec = fig.add_gridspec(ncols=4, nrows=1)
        ax1 = fig.add_subplot(spec[0, 0:3])
        ax2 = fig.add_subplot(spec[0, 3])
        latencies = []
        for soa in param["soas"]:
            evks = []
            lats = []
            vals = []
            # Loop through each subject:
            for sub in subjects_epochs.keys():
                # Average the data across both eyes:
                data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, task])], axis=1)
                # Remove any trials containing Nan:
                data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                # Remove any trials containing Nan:
                data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                evk = np.mean(data, axis=0)
                # Find the time at % peak:
                ind, val = max_percentage_index(evk, 90)
                lats.append(subjects_epochs[sub].times[ind])
                vals.append(val)
                latencies.append(pd.DataFrame({
                    "sub_id": sub,
                    "SOA_lock": "onset",
                    "SOA": soa,
                    "SOA_float": float(soa),
                    "latency": subjects_epochs[sub].times[ind],
                    "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                    "value": val
                }, index=[0]))
                evks.append(evk)
            # Plot the evoked pupil response:
            plot_ts_ci(np.array(evks), epochs.times, ev.colors["soa_onset_locked"][soa], plot_ci=False, ax=ax1, label=soa)
            # Plot the latency:
            mean_lat = np.mean(lats)
            ax1.vlines(x=float(soa), ymin=-0.02, ymax=-0.01, linestyle="-",
                       color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
            ax1.vlines(x=mean_lat, ymin=-0.02,
                       ymax=np.mean(np.array(evks), axis=0)[np.argmin(np.abs(epochs.times - mean_lat))],
                       linestyle="--",
                       color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
        ax1.set_ylim([-0.02, 0.09])
        ax1.legend()
        ax1.set_xlabel("Time (sec.)")
        ax1.set_ylabel("Pupil size (norm.)")
        # Plot the latency as a function of SOA:
        latencies = pd.concat(latencies).reset_index(drop=True)
        plot_within_subject_boxplot(latencies, "sub_id", "SOA_float", "latency_aud",
                                    positions="SOA_float", ax=ax2, cousineau_correction=False, title="", xlabel="SOA",
                                    ylabel=r'$\tau_{\mathrm{audio}}$', xlim=[-0.1, 0.6], width=0.1,
                                    face_colors=[ev.colors["soa_onset_locked"][soa] for soa in param["soas"]])
        ax2.set_ylim([0, 1.5])
        plt.suptitle("Pupil peak latency")
        fig.savefig(Path(save_dir, "pupil_latency_titr_{}-{}.svg".format(lock, task)), transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "pupil_latency_titr_{}-{}.png".format(lock, task)), transparent=True, dpi=300)
        plt.close()


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX118", "SX116", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\03-pupil_amplitude_parameters.json ")
    pupil_latency(parameters, subjects_list)
