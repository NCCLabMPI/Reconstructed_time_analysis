import os
import mne
import json
from pathlib import Path
from eye_tracker.preprocessing_helper_function import (extract_eyelink_events, epoch_data, dilation_speed_rejection,
                                                       interpolate_pupil, set_pupil_nans, remove_around_gap,
                                                       trend_line_departure, remove_bad_epochs)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import environment_variables as ev


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
    raw_file = Path(ev.bids_root, "sub-" + subject, "ses-" + session, data_type,
                    "sub-{}_ses-{}_task-{}_{}-raw.fif".format(subject, session, task, data_type))
    raw = mne.io.read_raw_fif(raw_file, verbose="WARNING")
    # Plot the data:
    # raw_ds = raw.copy().resample(100, npad="auto")
    # Remove the annotations:
    # raw_ds.annotations.delete(np.arange(len(raw_ds.annotations.description)))
    # raw_ds.plot(block=True)
    # Convert the annotations to event for epoching:
    print('Creating annotations')
    events_from_annot, event_dict = mne.events_from_annotations(raw, verbose="ERROR")

    # =============================================================================================
    # Loop through the preprocessing steps:
    for step in preprocessing_steps:
        # Extract the parameters of the current step:
        step_param = param[step]
        # Performing the cleaning:
        if step == "set_pupil_nans":
            raw = set_pupil_nans(raw,
                                 eyes=step_param["eyes"])
        # Performing the cleaning:
        if step == "dilation_speed_rejection":
            raw = dilation_speed_rejection(raw, threshold_factor=step_param["threshold_factor"],
                                           eyes=step_param["eyes"])
        if step == "remove_around_gap":
            raw = remove_around_gap(raw, gap_duration_s=step_param["gap_duration_s"],
                                    eyes=step_param["eyes"])
        if step == "interpolate_pupil":
            raw = interpolate_pupil(raw, eyes=step_param["eyes"], kind=step_param["kind"])
        if step == "trend_line_departure":
            raw = trend_line_departure(raw, threshold_factor=step_param["threshold_factor"],
                                       eyes=step_param["eyes"], window_length_s=step_param["window_length_s"],
                                       polyorder=step_param["polyorder"], n_iter=step_param["n_iter"])

        # Performing blinks, saccades and fixaction extraction:
        if step in ["extract_blinks", "extract_saccades", "extract_fixation"]:
            print("Extracting the {} from the annotation".format(step_param["description"]))
            raw = extract_eyelink_events(raw, step_param["description"])

        # Print the proportion of NaN in the data:
        total_nan_proportion = np.mean(
            [np.sum(np.isnan(raw.get_data(picks=["LPupil"]))) / raw.get_data().shape[-1],
             np.sum(np.isnan(raw.get_data(picks=["RPupil"]))) / raw.get_data().shape[-1]])

        if step == "epochs":
            # Looping through each of the different epochs file to create:
            for epoch_name in step_param.keys():
                print(epoch_name)
                epochs = epoch_data(raw, events_from_annot, event_dict, **step_param[epoch_name])

                if "remove_bad_epochs" in preprocessing_steps:
                    epochs, proportion_rejected_trials = remove_bad_epochs(epochs,
                                                                           nan_proportion_thresh=
                                                                           param["remove_bad_epochs"][
                                                                               "nan_proportion_thresh"])

                # Plot the epochs:
                if "discard_bad_subjects" in preprocessing_steps:
                    if (proportion_rejected_trials > param["discard_bad_subjects"]["bad_trials_threshold"] or
                            np.min(total_nan_proportion) > param["discard_bad_subjects"]["nan_threshold"]):
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
                if "extract_blinks" in preprocessing_steps:
                    # Plot the blinks rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="Lblink")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="Rblink")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_blinks.png".format(subject, session, task,
                                                                                     data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()
                if "extract_saccades" in preprocessing_steps:
                    # Plot the blinks rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="Lsaccade")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="Rsaccade")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_saccades.png".format(subject, session, task,
                                                                                       data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()
                if "extract_fixation" in preprocessing_steps:
                    # Plot the blinks rate:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="Lfixation")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="Rfixation")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_fixation.png".format(subject, session, task,
                                                                                       data_type, epoch_name)
                    plt.savefig(Path(save_root, file_name))
                    plt.close()

            return total_nan_proportion, proportion_rejected_trials


if __name__ == "__main__":
    # The following subjects have the specified issues:
    # SX101: differences in sampling rate due to experiment program issues
    # SX104: missing files
    # SX117: no eyetracking data
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    parameters_file = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\01-preprocessing_parameters.json ")
    # Create a data frame to save the summary of all subjects:
    preprocessing_summary = []
    for sub in subjects_list:
        print("Preprocessing subject {}".format(sub))
        total_nan, rejected_trials = preprocessing(sub, parameters_file)
        # Append to the summary:
        preprocessing_summary.append(pd.DataFrame({
            "subject": sub,
            "total_nan": total_nan,
            "rejected_trials": rejected_trials,
            "valid_flag": True if rejected_trials < 0.5 and np.min(total_nan) < 0.5 else False
        }, index=[0]))
    # Concatenate the data frame:
    preprocessing_summary = pd.concat(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants.csv"))
