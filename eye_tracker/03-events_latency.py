import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling, get_event_ts
from eye_tracker.plotter_functions import plot_within_subject_boxplot
from eye_tracker.plot_events_latencies import plot_events_latencies
import pandas as pd
import environment_variables as ev
import os

# Set the font size:
plt.rcParams.update({'font.size': 22})


def events_latency(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    events_latencies = pd.DataFrame()
    # Loop through each subject:
    for sub in subjects:
        print(sub)
        # ===========================================
        # Data loading:
        # Load the epochs:
        root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + param["session"],
                    param["data_type"])
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, param["session"], param["task"],
                                                                      param["data_type"],
                                                                      param["epoch_name"])
        epochs = mne.read_epochs(Path(root, file_name))
        # Extract the relevant conditions:
        epochs = epochs[["non-target", "irrelevant"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        # epochs.decimate(10)
        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # ===========================================
        # Preprocessing
        # Baseline correction:
        if param["baseline"] is not None:
            baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract the data:
                        if lock == 'onset':
                            data = epochs.copy()["/".join([task_rel, duration,
                                                           lock, soa])].crop(float(soa),
                                                                             float(soa) + 1).get_data()
                        else:
                            data = epochs.copy()["/".join([task_rel, duration,
                                                           lock, soa])].crop(float(soa) + float(duration),
                                                                             float(soa) + 1 +
                                                                             float(duration)).get_data()
                        # Reject any  event that was found only in 1 eye:
                        data = np.logical_and(data[:, 0, :], data[:, 1, :])
                        times = np.arange(0, 1 + 1 / epochs.info['sfreq'], 1 / epochs.info['sfreq'])
                        # Extract the time stamp of the event:
                        onset_ts = get_event_ts(data, times)
                        # Extract only the first saccade in the segment:
                        onset_ts = [trial[0] for trial in onset_ts if len(trial) > 0]
                        # Add to data frame using pd.concat:
                        events_latencies = pd.concat([events_latencies,
                                                      pd.DataFrame({"subject": [sub] * len(onset_ts),
                                                                    "task_relevance": [task_rel] * len(onset_ts),
                                                                    "duration": [duration] * len(onset_ts),
                                                                    "lock": [lock] * len(onset_ts),
                                                                    "soa": [soa] * len(onset_ts),
                                                                    "event_latency": onset_ts})])
    events_latencies = events_latencies.reset_index(drop=True)
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", param["event_type"], "group_level", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    events_latencies.to_csv(Path(save_dir, "events_latencies.csv"))

    # Plot events latencies:
    # Onset locked across durations:
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[14, 14])
    for task_rel_i, task_rel in enumerate(param["task_relevance"]):
        # Extract the data of this particular condition:
        df = events_latencies.loc[(events_latencies["task_relevance"] == task_rel)
                                  & (events_latencies["lock"] == 'onset'), ["subject", "soa", "event_latency"]]
        plot_within_subject_boxplot(df, 'subject', 'soa', 'event_latency',
                                    positions='soa', ax=ax[task_rel_i], cousineau_correction=True,
                                    title="{}".format(task_rel, 'Onset'),
                                    xlabel="SOA (s)", ylabel="{} latency (s)".format(param["event_type"]),
                                    xlim=[-0.1, 0.566],
                                    face_colors=list(ev.colors['soa'].values()))

    plt.suptitle("{} locked {} latency".format('Onset', param["event_type"]))
    plt.tight_layout()
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", param["event_type"], "group_level", "figures")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the figure:
    plt.savefig(Path(save_dir, "{}_locked_{}_latency.png".format('Onset', param["event_type"])))
    plt.savefig(Path(save_dir, "{}_locked_{}_latency.svg".format('Onset', param["event_type"])))
    plt.close(fig)

    # Separetely for each duration:
    for dur in param["durations"]:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[14, 14])
        for task_rel_i, task_rel in enumerate(param["task_relevance"]):
            # Extract the data of this particular condition:
            df = events_latencies.loc[(events_latencies["task_relevance"] == task_rel)
                                      & (events_latencies["duration"] == dur)
                                      & (events_latencies["lock"] == 'onset'), ["subject", "soa", "event_latency"]]
            plot_within_subject_boxplot(df, 'subject', 'soa', 'event_latency',
                                        positions='soa', ax=ax[task_rel_i], cousineau_correction=True,
                                        title="{}-{}".format(task_rel, 'Onset'),
                                        xlabel="SOA (s)", ylabel="{} latency (s)".format(param["event_type"]),
                                        xlim=[-0.1, 0.566], face_colors=list(ev.colors['soa'].values()))

        plt.suptitle("{}-{} locked {} latency".format('Onset', dur, param["event_type"]))
        plt.tight_layout()
        # Create the save directory:
        save_dir = Path(ev.bids_root, "derivatives", param["event_type"], "group_level", "figures")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # Save the figure:
        plt.savefig(Path(save_dir, "{}_locked_{}_{}_latency.png".format('Onset', dur, param["event_type"])))
        plt.savefig(Path(save_dir, "{}_locked_{}_{}_latency.svg".format('Onset', dur, param["event_type"])))
        plt.close(fig)

    # Offset locked:
    offset_latencies = events_latencies.loc[events_latencies["lock"] == "offset"]
    offset_latencies["soa"] = offset_latencies["soa"].astype(float) + offset_latencies["duration"].astype(float)
    for task_rel_i, task_rel in enumerate(param["task_relevance"]):
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[14, 14])
        # Extract the data of this particular condition:
        for dur_i, dur in enumerate(param["durations"]):
            df = offset_latencies.loc[(offset_latencies["task_relevance"] == task_rel)
                                      & (offset_latencies["duration"] == dur)
                                      & (offset_latencies["lock"] == 'offset'), ["subject", "soa", "event_latency"]]
            plot_within_subject_boxplot(df, 'subject', 'soa', 'event_latency',
                                        positions='soa', ax=ax[dur_i], cousineau_correction=True,
                                        title="{}".format(dur),
                                        xlabel="SOA (s)", ylabel="{} latency (s)".format(param["event_type"]),
                                        xlim=[np.min(df["soa"].to_numpy()) - 0.1,
                                              np.max(df["soa"].to_numpy()) + 0.1], width=0.1,
                                        face_colors=list(ev.colors['soa'].values()))
        ax[0].set_xlabel("SOA (sec)")
        ax[0].set_ylabel("Cousineau Morey Corrected {} latency".format(param["event_type"]))
        ax[1].set_ylabel("")
        ax[2].set_ylabel("")
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[2].spines['right'].set_visible(False)

        plt.suptitle("{} locked {} latency".format('Offset', param["event_type"]))
        plt.tight_layout()
        # Create the save directory:
        save_dir = Path(ev.bids_root, "derivatives", param["event_type"], "group_level", "figures")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # Save the figure:
        plt.savefig(Path(save_dir, "{}_locked_{}_{}_latency.png".format('offset', task_rel, param["event_type"])))
        plt.savefig(Path(save_dir, "{}_locked_{}_{}_latency.svg".format('offset', task_rel, param["event_type"])))
        plt.close(fig)

    plot_events_latencies(subjects, parameters_file)


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    # Perform the blinks latency analysis
    events_latency(r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
                   r"\03-blinking_latency_parameters.json ", subjects_list)
    # Perform the saccades latency analysis
    events_latency(r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
                   r"\03-saccades_latency_parameters.json ", subjects_list)
