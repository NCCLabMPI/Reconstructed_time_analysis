import json
import mne
import mne_bids
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def integrate_log_events(events, log_file, event_id, sfreq=1000, trials_duration=2.75):
    description = []
    onset = []
    duration = []
    evt_ctr = 0
    # Loop through the log file:
    for trial_i in range(log_file.shape[0]):
        # Extract the current trial infos:
        soa = log_file['soa'].iloc[trial_i]
        onset_soa = log_file['onset_SOA'].iloc[trial_i]
        soa_lock = log_file['soa_lock'].iloc[trial_i]
        task = log_file['task'].iloc[trial_i]
        task_relevance = log_file['task_relevance'].iloc[trial_i]
        dur = log_file['duration'].iloc[trial_i]
        category = log_file['category'].iloc[trial_i]
        identity = log_file['identity'].iloc[trial_i]
        orientation = log_file['orientation'].iloc[trial_i]
        pitch = log_file['pitch'].iloc[trial_i]
        # Depending on whether the task was visual first or auditory first, the first trigger will be different:
        if task == 'visual_first':
            # Extract the current trigger:
            trig = events[evt_ctr, 2]
            # Extract event id:
            trig_id = [key for key, value in event_id.items() if value == trig][0]
            if trig_id != 'visual':
                raise ValueError('The current trigger is not a visual stimulus!')
            onset.append(events[evt_ctr, 0] / sfreq)
            duration.append(log_file['duration'].iloc[trial_i])
            # Create the description string:
            description.append("/".join(["visual_onset", task, task_relevance, str(dur), category, identity,
                                         orientation, str(soa), soa_lock, str(onset_soa), str(pitch)]))
            # Increment the event counter:
            evt_ctr += 1

            # If the soa is not of 0, then we should have a trigger for everything:
            if soa != 0:
                for i in range(3):
                    # Extract the current trigger:
                    trig = events[evt_ctr, 2]
                    # Extract event id:
                    trig_id = [key for key, value in event_id.items() if value == trig][0]
                    if trig_id == 'audio':
                        evt_type = 'auditory_onset'
                    elif trig_id == 'fixation':
                        evt_type = 'fixation'
                    elif trig_id == 'jitter':
                        evt_type = 'jitter'
                    else:
                        raise ValueError('The current trigger is not a visual stimulus!')
                    onset.append(events[evt_ctr, 0] / sfreq)
                    duration.append(log_file['duration'].iloc[trial_i])
                    description.append(
                        "/".join([evt_type, task, task_relevance, str(dur), category, identity,
                                  orientation, str(soa), soa_lock, str(onset_soa), str(pitch)]))
                    evt_ctr += 1
            else:
                # If the soa is 0, then we should have a trigger for the fixation next:
                trig = events[evt_ctr, 2]
                # Extract event id:
                trig_id = [key for key, value in event_id.items() if value == trig][0]
                if trig_id != 'fixation':
                    raise ValueError('The current trigger is not a fixation stimulus!')
                # Recreate the sound trigger:
                if soa_lock == 'onset':
                    onset_reconstructed = (events[evt_ctr - 1, 0] / sfreq) + \
                                          (log_file['aud_stim_time'].iloc[trial_i] -
                                           log_file['vis_stim_time'].iloc[trial_i])
                else:
                    onset_reconstructed = (events[evt_ctr - 1, 0] / sfreq) + \
                                          (log_file['aud_stim_time'].iloc[trial_i] -
                                           log_file['fix_time'].iloc[trial_i])
                onset.append(onset_reconstructed)
                duration.append(80)
                # Create the description string:
                description.append("/".join(["auditory_onset", task, task_relevance, str(dur), category,
                                             identity, orientation, str(soa), soa_lock, str(onset_soa), str(pitch)]))
                # Add the fixation event:
                onset.append(events[evt_ctr, 0] / sfreq)
                duration.append(trials_duration - dur)
                # Create the description string:
                description.append("/".join(["fixation", task, task_relevance, str(dur), category, identity,
                                             orientation, str(soa), soa_lock, str(onset_soa), str(pitch)]))
                # Increment the event counter:
                evt_ctr += 1

                # The final trigger of this trial should be the jitter:
                trig = events[evt_ctr, 2]
                # Extract event id:
                trig_id = [key for key, value in event_id.items() if value == trig][0]
                if trig_id != 'jitter':
                    raise ValueError('The current trigger is not a jitter stimulus!')
                onset.append(events[evt_ctr, 0] / sfreq)
                duration.append(trials_duration - dur)
                # Create the description string:
                description.append("/".join(["fixation", task, task_relevance, str(dur), category, identity,
                                             orientation, str(soa), soa_lock, str(onset_soa), str(pitch)]))
                # Increment the event counter:
                evt_ctr += 1
    # Convert everything to mne annotations:
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)

    return annotations


def validate_triggers(events, log_file):
    # In our experiment, we send 3-4 triggers per trial. Trials with SOA 0 only have 3 triggers, 4 otherwise:
    n_trials = log_file.shape[0]
    n_soa_0_trials = np.sum(log_file['soa'] == 0)
    # Compute the expected number of triggers:
    n_expected_triggers = n_trials * 4 - n_soa_0_trials
    # Compare to the actual number of triggers:
    n_actual_triggers = events.shape[0]
    if n_expected_triggers != n_actual_triggers:
        raise ValueError(f"Expected {n_expected_triggers} triggers, but found {n_actual_triggers}.")
    else:
        print(f"Found {n_actual_triggers} triggers.")
    return None


def quality_checks_plots(raw, events, event_id, subject, session, run, datatype, log_files, save_path=None,
                         show_plots=False):
    """
    """
    # Check if the save path exist on the computer:
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.exists():
            print("WARNING: The save path does not exist. Creating it now.")
            save_path.mkdir(parents=True)
    # Plot the events:
    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id, show=show_plots,
        on_missing='warn'
    )
    fig.subplots_adjust(right=0.7)  # make room for legend
    # Save the figure:
    if save_path is not None:
        fig.savefig(save_path / f"{subject}_{session}_{run}_{datatype}_events.png")

    # Plot the stimuli duration determined by the triggers:
    visual_onsets = events[events[:, 2] == event_id['visual'], 0]
    fixation_onsets = events[events[:, 2] == event_id['fixation'], 0]
    # Convert each to seconds:
    visual_onsets = visual_onsets / raw.info['sfreq']
    fixation_onsets = fixation_onsets / raw.info['sfreq']
    diff = fixation_onsets - visual_onsets
    # Plot the difference:
    fig, ax = plt.subplots()
    ax.hist(diff, bins=100)
    ax.set_xlabel("Stimulus duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"sub-{subject}_ses-{session}_run-{run}_{datatype}_stimulus_duration")
    # Save the figure:
    if save_path is not None:
        fig.savefig(save_path / f"{subject}_{session}_{run}_{datatype}_trigger_stimulus_duration.png")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Plot the comparison between the log file and the triggers:
    visual_onsets = log_files['vis_stim_time'].to_numpy()
    fixation_onsets = log_files['fix_time'].to_numpy()
    logs_diff = fixation_onsets - visual_onsets
    # Plot the comparison between the log files and the triggers:
    fig, ax = plt.subplots()
    ax.plot(range(len(diff)), diff, label="triggers")
    ax.plot(range(len(logs_diff)), logs_diff, label="Log file")
    ax.set_xlabel("Intervals (s)")
    ax.set_ylabel("Stimulus duration (s)")
    ax.set_title(f"sub-{subject}_ses-{session}_run-{run}_{datatype}_stimulus_duration")
    ax.legend()
    # Save the figure:
    if save_path is not None:
        fig.savefig(save_path / f"{subject}_{session}_{run}_{datatype}_stimulus_duration_comparison.png")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Plot the distribution of the difference:
    diff_of_diff = diff - logs_diff
    fig, ax = plt.subplots()
    ax.hist(diff_of_diff, bins=100)
    ax.set_xlabel("Difference (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"sub-{subject}_ses-{session}_run-{run}_{datatype}_stimulus_duration_difference")
    # Save the figure:
    if save_path is not None:
        fig.savefig(save_path / f"{subject}_{session}_{run}_{datatype}_stimulus_duration_difference.png")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return None


def split_logs(log_file, n_runs=4, columns="block"):
    """
    This function splits the log file into chunks of 4 runs.
    :param log_file: pandas data frame to split in chunks according to a specific column. That column must specify a
    range
    :param n_runs: (int) number of runs to split the log file in
    :param columns: column to split the log file according to
    :return: a list of pandas data frames, each containing a chunk of the log file
    """
    # Extract the block numbers:
    blocks = log_file[columns].to_numpy()
    # Get the unique block numbers:
    unique_blocks = np.unique(blocks)
    # We have n unique blocks, so we split them into n_runs chunks:
    blocks_split = np.array_split(unique_blocks, np.ceil(len(unique_blocks) / n_runs))
    log_file_split = []
    # Loop through each block:
    for i, block in enumerate(blocks_split):
        # Get the indices of the current block:
        idx = np.isin(blocks, block)
        # Get the current block:
        log_file_split.append(log_file.loc[idx, :])
    return log_file_split


def get_meg_file(raw_root, subject, session, task, run, subfolder="meg"):
    """
    This function locates the right MEG file from a list of files. The name of the MEG files contains a date which
    is inconscequential to the experiment, so it is not included in the search. The file name is of the following
    structure: sub-SX103_MPIEA0357.ReconTime_20230718_task-visual_first_ses-1_run-1_meg.ds, but we only want to parse
    by sub, task, ses and run.
    :param raw_root:
    :param subject:
    :param session:
    :param task:
    :param run:
    :param subfolder:
    :return:
    """
    # Get the list of file found in the raw root:
    files = list(Path(raw_root, subfolder).glob("*.ds"))
    # Matching files:
    matching_files = []
    for file in files:
        # Create dict from the current file name, extracting the subject, session, task and run:
        file_dict = dict([tuple(x.split("-")) for x in file.name.split("_") if "-" in x])
        if "sub" not in file_dict.keys() or "ses" not in file_dict.keys() or "task" not in file_dict.keys() or \
                "run" not in file_dict.keys():
            continue
        # Check that the subject, session, task and run match the input:
        if file_dict["sub"] == subject and file_dict["ses"] == session and file_dict["task"] == task and \
                file_dict["run"] == run:
            matching_files.append(file)
    # Check that only one file was found:
    if len(matching_files) > 1:
        raise ValueError(f"More than one file was found for subject {subject}, session {session}, task {task} and "
                         f"run {run}.")
    elif len(matching_files) == 0:
        raise ValueError(f"No file was found for subject {subject}, session {session}, task {task} and "
                         f"run {run}.")
    else:
        return matching_files[0]
