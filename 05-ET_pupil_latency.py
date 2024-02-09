import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from helper_function.helper_general import baseline_scaling, max_percentage_index, equate_epochs_events
from helper_function.helper_plotter import plot_pupil_latency
import environment_variables as ev
import pandas as pd

# Set the font size:
plt.rcParams.update({'font.size': 14})


def pupil_latency(parameters_file, subjects, session="1", task="prp"):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Load all subjects data:
    subjects_epochs = {sub: None for sub in subjects}
    # Create the directory to save the results in:
    save_dir = Path(ev.bids_root, "derivatives", "pupil_latency", task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # The soas differ between the experiments, therefore, extracting them directly from the epochs objects:
    soas = []
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
        # Extract the soas:
        soas = list(epochs.metadata["SOA"].unique())
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
    for soa in soas:
        for task in param["task_relevance"]:
            for duration in param["duration"]:
                for lock in param["lock"]:
                    for sub in subjects_epochs.keys():
                        # Average the data across both eyes:
                        if lock == "offset":
                            # Extract the data of the particular condition and crop the data:
                            sub_epochs = subjects_epochs[sub].copy().crop(float(duration) + float(soa),
                                                                          subjects_epochs[sub].times[-1])[
                                "/".join([soa, task, duration, lock])]
                            # Crop the data:
                            data = np.nanmean(sub_epochs, axis=1)
                        else:
                            # Extract the data of the particular condition:
                            sub_epochs = subjects_epochs[sub].copy().crop(float(soa), subjects_epochs[sub].times[-1])[
                                "/".join([soa, task, duration, lock])]
                            data = np.nanmean(sub_epochs, axis=1)
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                        evk = np.mean(data, axis=0)
                        ind, val = max_percentage_index(evk, 90)
                        # Find the time at % peak:
                        if lock == "offset":
                            soa_locked = float(soa) + float(duration)
                        else:
                            soa_locked = float(soa)
                        latency = sub_epochs.times[ind]
                        latency_aud = sub_epochs.times[ind] - soa_locked

                        latencies_lmm.append(pd.DataFrame({
                            "sub_id": sub,
                            "SOA": soa,
                            "task": task,
                            "duration": duration,
                            "lock": lock,
                            "SOA_float": float(soa),
                            "SOA_locked": soa_locked,
                            "latency": latency,
                            "latency_aud": latency_aud,
                            "amplitude": val
                        }, index=[0]))
    # Convert to data frame:
    latencies_lmm = pd.concat(latencies_lmm).reset_index(drop=True)
    # Save the peak latencies:
    latencies_lmm.to_csv(Path(save_dir, "pupil_peak_latencies.csv"))

    # ==================================================================================================================
    # Plotting of SOAs comparisons:
    # ===============================================================================================
    # Loop through locks:
    for lock in param["lock"]:
        if lock == "onset":
            boxplot_ylim = [0, 1.5]
        else:
            boxplot_ylim = None
        # ===========================================================
        # Per SOA:
        if lock == "onset":  # For the offset locked trials, we need to plot separately per stimuli durations:
            evks = {soa: [] for soa in soas}
            for soa in soas:
                # Loop through each subject:
                for sub in subjects_epochs.keys():
                    # Average the data across both eyes:
                    data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock])], axis=1)
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                    evk = np.mean(data, axis=0)
                    evks[soa].append(evk)
            # Compute mean latency per subject and SOA:
            latencies = latencies_lmm[latencies_lmm["lock"] == lock].groupby(['sub_id', 'SOA', 'SOA_locked'],
                                                                             as_index=False)[
                ["latency", "latency_aud"]].mean()
            fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_{}_locked".format(lock)],
                                     boxplot_ylim=boxplot_ylim)
            fig.savefig(Path(save_dir, "pupil_latency_{}.svg".format(lock)), transparent=True, dpi=300)
            fig.savefig(Path(save_dir, "pupil_latency_{}.png".format(lock)), transparent=True, dpi=300)
            plt.close()

            # ===========================================================
            # Separately for each task relevance condition:
            for task in param["task_relevance"]:
                evks = {soa: [] for soa in soas}
                for soa in soas:
                    # Loop through each subject:
                    for sub in subjects_epochs.keys():
                        # Average the data across both eyes:
                        data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, task])], axis=1)
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                        evk = np.mean(data, axis=0)
                        evks[soa].append(evk)
                # Compute mean latency per subject and SOA:
                latencies = latencies_lmm[(latencies_lmm["lock"] == lock) &
                                          (latencies_lmm["task"] == task)].groupby(
                    ['sub_id', 'SOA', 'SOA_locked'],
                    as_index=False)[["latency",
                                     "latency_aud"]].mean()
                fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_{}_locked".format(lock)],
                                         boxplot_ylim=boxplot_ylim)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, task)), transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, task)), transparent=True, dpi=300)
                plt.close()

        # ===========================================================
        # Separately for each durations:
        for duration in param["duration"]:
            evks = {soa: [] for soa in soas}
            for soa in soas:
                # Loop through each subject:
                for sub in subjects_epochs.keys():
                    # Average the data across both eyes:
                    data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, duration])], axis=1)
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                    # Remove any trials containing Nan:
                    data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                    evk = np.mean(data, axis=0)
                    evks[soa].append(evk)
            # Compute mean latency per subject and SOA:
            latencies = (
                latencies_lmm[(latencies_lmm["lock"] == lock) &
                              (latencies_lmm["duration"] == duration)].groupby(['sub_id', 'SOA', 'SOA_locked'],
                                                                               as_index=False)[["latency",
                                                                                                "latency_aud"]].mean())
            fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_{}_locked".format(lock)],
                                     boxplot_ylim=boxplot_ylim)
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, duration)),
                        transparent=True, dpi=300)
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, duration)),
                        transparent=True, dpi=300)
            plt.close()

        # ===========================================================
        # Separately for each durations and task relevance:
        for task in param["task_relevance"]:
            for duration in param["duration"]:
                evks = {soa: [] for soa in soas}
                for soa in soas:
                    # Loop through each subject:
                    for sub in subjects_epochs.keys():
                        data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, lock, duration, task])],
                                          axis=1)
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
                        # Remove any trials containing Nan:
                        data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
                        evk = np.mean(data, axis=0)
                        evks[soa].append(evk)
                # Compute mean latency per subject and SOA:
                latencies = (
                    latencies_lmm[(latencies_lmm["lock"] == lock) &
                                  (latencies_lmm["duration"] == duration) &
                                  (latencies_lmm["task"] == task)].groupby(['sub_id', 'SOA',
                                                                            'SOA_locked'],
                                                                           as_index=False)[["latency",
                                                                                            "latency_aud"]].mean())
                fig = plot_pupil_latency(evks, times, latencies, ev.colors["soa_{}_locked".format(lock)],
                                         boxplot_ylim=boxplot_ylim)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.svg".format(lock, task, duration)),
                            transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.png".format(lock, task, duration)),
                            transparent=True, dpi=300)
                plt.close()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\04-pupil_latency_parameters.json ")

    # Subjects lists:
    subjects_list_prp = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112",
                         "SX113", "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121", "SX123"]
    subjects_list_intro = ["SX101", "SX105", "SX106", "SX108", "SX109", "SX110", "SX113", "SX114",
                           "SX115", "SX116", "SX118"]
    # ==================================================================================
    # Introspection analysis:
    pupil_latency(parameters, subjects_list_intro, task="introspection", session=["2", "3"])

    # ==================================================================================
    # PRP analysis:
    pupil_latency(parameters, subjects_list_prp, task="prp", session="1")
