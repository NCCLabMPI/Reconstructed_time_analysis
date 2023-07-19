import json
import mne
import mne_bids
import numpy as np
from pathlib import Path
import pandas as pd
from bids_conversion_helper import split_logs, get_meg_file, validate_triggers, quality_checks_plots, \
    integrate_log_events


def bids_converter_handler(subject, session, task, run, config="bids_conversion_params.json",
                           show_events=False):
    """
    This function is a wrapper for the bids conversion function. It takes the raw data and the behavioral log files
    and converts them to a BIDS dataset.
    :param subject:
    :param session:
    :param task:
    :param run:
    :param config:
    :param show_events:
    :return:
    """
    # Load the config:
    with open(config, "r") as f:
        param = json.load(f)

    # Load the log file:
    log_path = Path(param["raw_root"], "beh", f"sub-{subject}", f"ses-{session}",
                    f"sub-{subject}_ses-{session}_run-{run}_task-{task}_events.csv")
    log_file = pd.read_csv(log_path, sep=',')
    # Parse the log file into chunks of 4 runs:
    log_files_list = split_logs(log_file, n_runs=param["n_runs_per_block"], columns=param["blk_column_name"])

    # Loop through each log file:
    for i, log_file in enumerate(log_files_list):
        print("=============================================================")
        print(f"Processing subject {subject}, session {session}, task {task} and run {i+1}")
        # Create the bids path for this run:
        bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task.split("_")[0], run=i+1,
                                      datatype="meg",
                                      root=param["bids_root"])
        bids_path.mkdir(exist_ok=True)
        # Load the raw file:
        raw_path = get_meg_file(param["raw_root"], subject, session, task.split("_")[0], str(i + 1), subfolder="meg")
        raw = mne.io.read_raw_ctf(raw_path, preload=False)
        # Extract the trigger channel only:
        raw_trig = raw.copy().pick_channels([param["trigger_channel"]])
        raw_trig.load_data()
        # Extract the events:
        events = mne.find_events(raw_trig, stim_channel=param["trigger_channel"])
        # Remove the response triggers for now:
        events = events[events[:, 2] != param["event_dict"]["response"], :]
        # Replace the events that are equal to 7 by a 3:
        events[events[:, 2] == 7, 2] = 3
        # Remove the 246th trigger:
        if i == 1:
            events = np.delete(events, 247, axis=0)
        if i == 2:
            events = np.delete(events, 153, axis=0)
        if i == 3:
            events = np.delete(events, 311, axis=0)
        # Check that the number of triggers matches the number of events:
        validate_triggers(events, log_file)
        # Plot the events:
        quality_checks_plots(raw_trig, events, param["event_dict"], subject, session, i+1, "meg", log_file,
                             save_path=bids_path.directory, show_plots=show_events)
        # Integrate the log file events with the triggers to create more sensical events:
        annotations = integrate_log_events(events, log_file, param["event_dict"],
                                           sfreq=raw_trig.info['sfreq'], trials_duration=param["trials_duration"])
        # Add the annotations to the raw:
        raw.set_annotations(annotations)
        # Save to BIDS:
        mne_bids.write_raw_bids(raw, bids_path,
                                overwrite=True, format='FIF')

        # Save the param file to the bids directory:
        param_path = Path(bids_path.directory, "sub-" + subject + "_ses-" + session + "_task-" + task + "_param.json")
        with open(param_path, "w") as f:
            json.dump(param, f, indent=4)


if __name__ == "__main__":
    bids_converter_handler("SX104", "1", "visual_first", "all", config="bids_conversion_param.json",
                           show_events=False)
