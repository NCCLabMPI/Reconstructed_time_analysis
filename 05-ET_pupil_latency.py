import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from helper_function.helper_general import (baseline_scaling, max_percentage_index, equate_epochs_events,
                                            reject_bad_epochs, format_drop_logs)
from helper_function.helper_plotter import plot_pupil_latency
import environment_variables as ev
import pandas as pd

# Set the font size:
plt.rcParams.update({'font.size': 14})


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
        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])
        subjects_epochs[sub] = epochs
    # Extract the times of the first subject (assuming that it is the same for all subjects, which it should be):
    times = subjects_epochs[subjects[0]].times

    # Plot the drop logs:
    drop_log_df = format_drop_logs({sub: subjects_epochs[sub].drop_log for sub in subjects_epochs.keys()})
    # Plot the drop log:
    # Extract the columns:
    cols = [col for col in drop_log_df.columns if col != "sub"]
    fig, ax = plt.subplots(figsize=[8.3, 8.3])
    ax.boxplot([drop_log_df[col].to_numpy() for col in cols], labels=cols)
    ax.axhline(param["drop_trials_threshold"], linestyle="--", color="r")
    ax.set_ylabel("Proportion dropped trials")
    ax.set_xlabel("Reason")
    plt.tight_layout()
    fig.savefig(Path(save_dir, "drop_log.svg"), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "drop_log.png"), transparent=True, dpi=300)
    plt.close()

    # Extract the subject that exceed the proportion of dropped trials
    drop_subjects = drop_log_df.loc[drop_log_df["total"] >= param["drop_trials_threshold"], "sub"].to_list()
    for sub in drop_subjects:
        del subjects_epochs[sub]

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
            pupil_size_ylim = [-50, 150]
        else:
            boxplot_ylim = None
            pupil_size_ylim = [-50, 50]
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
                                     boxplot_ylim=boxplot_ylim, pupil_size_ylim=pupil_size_ylim,
                                     figsize=[8.3, 11.7/2])
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
                                         boxplot_ylim=boxplot_ylim, pupil_size_ylim=pupil_size_ylim,
                                         figsize=[8.3, 11.7/2])
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, task)), transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, task)), transparent=True, dpi=300)
                plt.close()

        # ===========================================================
        # Separately for each durations:
        for dur_i, duration in enumerate(param["duration"]):
            if dur_i == 2:
                plot_legend = True
            else:
                plot_legend = False
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
                                     boxplot_ylim=boxplot_ylim, pupil_size_ylim=pupil_size_ylim,
                                     plot_legend=plot_legend,
                                     figsize=[8.3, 11.7/4])
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.svg".format(lock, duration)),
                        transparent=True, dpi=300)
            fig.savefig(Path(save_dir, "pupil_latency_{}-{}.png".format(lock, duration)),
                        transparent=True, dpi=300)
            plt.close()

        # ===========================================================
        # Separately for each durations and task relevance:
        for task in param["task_relevance"]:
            for dur_i, duration in enumerate(param["duration"]):
                if dur_i == 2:
                    plot_legend = True
                else:
                    plot_legend = False
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
                                         boxplot_ylim=boxplot_ylim, pupil_size_ylim=pupil_size_ylim,
                                         plot_legend=plot_legend,
                                         figsize=[8.3, 11.7/4])
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.svg".format(lock, task, duration)),
                            transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "pupil_latency_{}-{}-{}.png".format(lock, task, duration)),
                            transparent=True, dpi=300)
                plt.close()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\05-ET_pupil_latency_parameters.json")
    # ==================================================================================
    # Introspection analysis:
    task = "introspection"
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session=["2", "3"],
                  analysis_name="pupil_latency", reject_bad_trials=True)
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session=["2", "3"],
                  analysis_name="pupil_latency_no_rej", reject_bad_trials=False)

    # ==================================================================================
    # PRP analysis:
    task = "prp"
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session="1",
                  analysis_name="pupil_latency", reject_bad_trials=True)
    pupil_latency(parameters, ev.subjects_lists_et[task], task=task, session="1",
                  analysis_name="pupil_latency_no_rej", reject_bad_trials=False)
