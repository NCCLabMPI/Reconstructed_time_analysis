import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling, max_percentage_index
from eye_tracker.plotter_functions import plot_pupil_latency
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
    # Extract the times of the first subject (assuming that it is the same for all subjects, which it should be):
    times = subjects_epochs[subjects[0]].times

    # ==================================================================================================================
    # Create LMM tables:
    # ===========================================================
    latencies_lmm = []
    for soa in param["soas"]:
        for task in param["task_relevance"]:
            for duration in param["duration"]:
                for lock in param["lock"]:
                    for sub in subjects_epochs.keys():
                        # Average the data across both eyes:
                        data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, task, duration, lock])], axis=1)
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                        evk = np.mean(data, axis=0)
                        # Find the time at % peak:
                        ind, val = max_percentage_index(evk, 90)
                        latencies_lmm.append(pd.DataFrame({
                            "sub_id": sub,
                            "SOA": soa,
                            "task": task,
                            "duration": duration,
                            "lock": lock,
                            "SOA_float": float(soa),
                            "latency": subjects_epochs[sub].times[ind],
                            "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                            "amplitude": val
                        }, index=[0]))
    # Convert to data frame:
    latencies_lmm = pd.concat(latencies_lmm).reset_index(drop=True)
    # Save the peak latencies:
    latencies_lmm.to_csv(Path(save_dir, "pupil_peak_latencies.csv"))

    # ==================================================================================================================
    # Plotting of SOAs comparisons:
    # ===============================================================================================
    # Looping through onset and offset locked separately:
    for lock in param["lock"]:
        # ===========================================================
        # Per SOA:
        latencies = []
        evks = {soa: [] for soa in param["soas"]}
        for soa in param["soas"]:
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
                # Package in a dictionary:
                latencies.append(pd.DataFrame({
                    "sub_id": sub,
                    "SOA_lock": "onset",
                    "SOA": soa,
                    "SOA_float": float(soa),
                    "latency": subjects_epochs[sub].times[ind],
                    "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                    "amplitude": val
                }, index=[0]))
                evks[soa].append(evk)
        latencies = pd.concat(latencies).reset_index(drop=True)
        fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_onset_locked"])
        fig.savefig(Path(save_dir, "pupil_latency_{}.svg".format(lock)), transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "pupil_latency_{}.png".format(lock)), transparent=True, dpi=300)
        plt.close()

        # ===========================================================
        # Separately for each task relevance condition:
        for task in param["task_relevance"]:
            latencies = []
            evks = {soa: [] for soa in param["soas"]}
            for soa in param["soas"]:
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
                    # Package in a dictionary:
                    latencies.append(pd.DataFrame({
                        "sub_id": sub,
                        "SOA_lock": "onset",
                        "SOA": soa,
                        "SOA_float": float(soa),
                        "latency": subjects_epochs[sub].times[ind],
                        "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                        "amplitude": val
                    }, index=[0]))
                    evks[soa].append(evk)
            latencies = pd.concat(latencies).reset_index(drop=True)
            fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_onset_locked"])
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, task)), transparent=True, dpi=300)
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, task)), transparent=True, dpi=300)
            plt.close()

        # ===========================================================
        # Separately for each durations:
        for duration in param["duration"]:
            latencies = []
            evks = {soa: [] for soa in param["soas"]}
            for soa in param["soas"]:
                # Loop through each subject:
                for sub in subjects_epochs.keys():
                    # Average the data across both eyes:
                    data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, duration])], axis=1)
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                    evk = np.mean(data, axis=0)
                    # Find the time at % peak:
                    ind, val = max_percentage_index(evk, 90)
                    # Package in a dictionary:
                    latencies.append(pd.DataFrame({
                        "sub_id": sub,
                        "SOA_lock": "onset",
                        "SOA": soa,
                        "SOA_float": float(soa),
                        "latency": subjects_epochs[sub].times[ind],
                        "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                        "amplitude": val
                    }, index=[0]))
                    evks[soa].append(evk)
            latencies = pd.concat(latencies).reset_index(drop=True)
            fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_onset_locked"])
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, duration)),
                        transparent=True, dpi=300)
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, duration)),
                        transparent=True, dpi=300)
            plt.close()

        # ===========================================================
        # Separately for each durations and task relevance:
        for task in param["task_relevance"]:
            for duration in param["duration"]:
                latencies = []
                evks = {soa: [] for soa in param["soas"]}
                for soa in param["soas"]:
                    # Loop through each subject:
                    for sub in subjects_epochs.keys():
                        # Average the data across both eyes:
                        data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, duration, task])], axis=1)
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                        evk = np.mean(data, axis=0)
                        # Find the time at % peak:
                        ind, val = max_percentage_index(evk, 90)
                        # Package in a dictionary:
                        latencies.append(pd.DataFrame({
                            "sub_id": sub,
                            "SOA_lock": "onset",
                            "SOA": soa,
                            "SOA_float": float(soa),
                            "latency": subjects_epochs[sub].times[ind],
                            "latency_aud": subjects_epochs[sub].times[ind] - float(soa),
                            "amplitude": val
                        }, index=[0]))
                        evks[soa].append(evk)
                latencies = pd.concat(latencies).reset_index(drop=True)
                fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_onset_locked"])
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.svg".format(lock, task, duration)),
                            transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.png".format(lock, task, duration)),
                            transparent=True, dpi=300)
                plt.close()


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX118", "SX116", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\04-pupil_latency_parameters.json ")
    pupil_latency(parameters, subjects_list)
