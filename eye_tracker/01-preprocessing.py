import mne
from mne.viz.eyetracking import plot_gaze
import json
from pathlib import Path
from eye_tracker.preprocessing_helper_function import (extract_eyelink_events, epoch_data, dilation_speed_rejection,
                                                       trend_line_departure, remove_bad_epochs, show_bad_segments,
                                                       load_raw_eyetracker, compute_proportion_bad, add_logfiles_info,
                                                       gaze_to_dva)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import environment_variables as ev
import os

DEBUG = False
show_interpolated = False


def preprocessing(subject, parameters):
    """
    This function preprocesses the eyetracking data, using several MNE key functionalities for handling the data
    :param subject: (string) name of the subject to process. Note: do not include the sub-!
    :param parameters: (string) parameter json file
    :return: None: saves the epochs to file
    """
    # First, load the parameters:
    with open(parameters) as json_file:
        param = json.load(json_file)
    # Extract the info about the session:
    session = param["session"]
    task = param["task"]
    data_type = param["data_type"]
    preprocessing_steps = param["preprocessing_steps"]

    # =============================================================================================
    # Load the data:
    files_root = Path(ev.bids_root, "sub-" + subject, "ses-" + session, data_type)
    # Load the eyetracker data and associated files:
    logs_list, raws_list, calibs_list, screen_size, screen_distance, screen_res = (
        load_raw_eyetracker(files_root, subject, session, task, ev.raw_root,
                            param["beh_file_name"],
                            param["epochs"]["visual_onset"]["metadata_column"],
                            'vis_onset', verbose=False, debug=DEBUG))
    # Concatenate the objects
    raw = mne.concatenate_raws(raws_list)
    # Remove the empty calibrations:
    calibs = list(filter(None, calibs_list))
    # Flatten the list:
    calibs = [item for items in calibs for item in items]
    # Concatenate the log files:
    log_df = pd.concat(logs_list).reset_index(drop=True)
    # raw.plot(block=True)
    # =============================================================================================
    # Loop through the preprocessing steps:
    for step in preprocessing_steps:
        # Extract the parameters of the current step:
        step_param = param[step]

        # Apply dilation speed filter:
        if step == "dilation_speed_rejection":
            raw = dilation_speed_rejection(raw, threshold_factor=step_param["threshold_factor"],
                                           eyes=step_param["eyes"],
                                           window_length_s=step_param["window_length_s"])
        # Apply trend line departure filter:
        if step == "trend_line_departure":
            raw = trend_line_departure(raw, threshold_factor=step_param["threshold_factor"],
                                       eyes=step_param["eyes"], window_length_s=step_param["window_length_s"],
                                       n_iter=step_param["n_iter"])
        # Interpolate the data:
        if step == "interpolate_blinks":
            # Extract the annotations:
            annotations = raw.annotations.copy()
            # Extract the bad descriptions:
            bad_descriptions = list(set([val for val in raw.annotations.description if "BAD_" in val]))
            # Interpolate
            mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer=step_param["buffer"],
                                                             match=bad_descriptions,
                                                             interpolate_gaze=step_param["interpolate_gaze"])
            # Add the annotations back in as we still want to keep track what was interpolated and what wasn't:
            raw.set_annotations(annotations)
            # Show where the data were interpolated:
            if show_interpolated:
                show_bad_segments(raw, "left", pad_sec=1)
                show_bad_segments(raw, "right", pad_sec=1)
        # Extract the eyelink events as channels (to keep them after the epoching):
        if step == "extract_eyelink_events":
            print("Extracting the {} from the annotation".format(step_param["events"]))
            # Loop through each event to extract:
            for evt in step_param["events"]:
                raw = extract_eyelink_events(raw, evt, eyes=step_param["eyes"])

        if step == "gaze_to_dva":
            raw = gaze_to_dva(raw, screen_size, screen_res, screen_distance, eyes=step_param["eyes"])

        # Print the proportion of NaN in the data:
        proportion_bad = compute_proportion_bad(raw, desc="BAD_", eyes=["left", "right"])

        if step == "epochs":
            # Looping through each of the different epochs file to create:
            for epoch_name in step_param.keys():
                print(epoch_name)
                # Convert the annotations to event for epoching:
                print('Creating annotations')
                events_from_annot, event_dict = mne.events_from_annotations(raw, verbose="ERROR",
                                                                            regexp=param["events_of_interest"][0])
                # Epoch the data:
                epochs = epoch_data(raw, events_from_annot, event_dict, **step_param[epoch_name])

                # Add the log file information to the metadata
                if len(param["log_file_columns"]) > 0:
                    epochs = add_logfiles_info(epochs, log_df, param["log_file_columns"])

                if "remove_bad_epochs" in preprocessing_steps:
                    epochs, proportion_rejected_trials = remove_bad_epochs(epochs,
                                                                           channels=param["remove_bad_epochs"][
                                                                               "channels"],
                                                                           bad_proportion_thresh=
                                                                           param["remove_bad_epochs"][
                                                                               "nan_proportion_thresh"])

                # Plot the epochs:
                if "discard_bad_subjects" in preprocessing_steps:
                    if (proportion_rejected_trials > param["discard_bad_subjects"]["bad_trials_threshold"] or
                            np.mean(proportion_bad) > param["discard_bad_subjects"]["nan_threshold"]):
                        print("Subject {} rejected due to bad epochs".format(subject))
                        continue

                # Save this epoch to file:
                save_root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + subject,
                                 "ses-" + session, data_type)
                if not os.path.isdir(save_root):
                    os.makedirs(save_root)
                # Generate the file name:
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(subject, session, task, data_type,
                                                                              epoch_name)
                # Save:
                epochs.save(Path(save_root, file_name), overwrite=True, verbose="ERROR")
                epochs.load_data()
                # Depending on whehter or no the events were extracted:
                if "extract_eyelink_events" in preprocessing_steps:
                    # Plot the blinks rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="BAD_blink_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="BAD_blink_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_blinks.png".format(subject, session, task,
                                                                                     data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()

                    # Plot the saccades rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="saccade_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="saccade_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_saccades.png".format(subject, session, task,
                                                                                       data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()

                    # Plot the fixation rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="fixation_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="fixation_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_fixation.png".format(subject, session, task,
                                                                                       data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()

                # Finally, plot the fixation maps:
                plot_gaze(epochs, width=1920, height=1080, show=False)
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_gaze.png".format(subject, session, task,
                                                                               data_type, epoch_name)
                plt.savefig(Path(save_root, file_name))
                plt.close()

                # Finally, plot the calibrations:
                if param["plot_calibration"]:
                    for calib_i, calib in enumerate(calibs):
                        print(f"Calibration: {calib_i}")
                        print(calib)
                        calib.plot(show=False)
                        file_name = "calibration-{}_task-{}.png".format(calib_i, task)
                        plt.savefig(Path(save_root, file_name))
                        plt.close()

            return np.mean(proportion_bad), proportion_rejected_trials


if __name__ == "__main__":
    # The following subjects have the specified issues:
    # SX101: differences in sampling rate due to experiment program issues
    # SX104: missing files
    # SX117: no eyetracking data
    # , "SX103", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
    #                      "SX114", "SX115", "SX116", "SX119", "SX120", "SX121"
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX119", "SX120", "SX121"]

    parameters_file = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\01-preprocessing_parameters_task-auditory.json ")
    # Create a data frame to save the summary of all subjects:
    preprocessing_summary = []
    for sub in subjects_list:
        print("Preprocessing subject {}".format(sub))
        total_nan, rejected_trials = preprocessing(sub, parameters_file)
        # Append to the summary:
        preprocessing_summary.append(pd.DataFrame({
            "subject": sub,
            "total_bad": total_nan,
            "rejected_trials": rejected_trials,
            "valid_flag": True if rejected_trials < 0.5 and np.min(total_nan) < 0.5 else False
        }, index=[0]))
    # Concatenate the data frame:
    preprocessing_summary = pd.concat(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants.csv"))
