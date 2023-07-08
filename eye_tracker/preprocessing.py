import os
import mne
import json
from pathlib import Path
from eye_tracker.preprocessing_helper_function import extract_eyelink_events, epoch_data
import matplotlib.pyplot as plt
import numpy as np


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
    bids_root = param["bids_root"]
    session = param["session"]
    task = param["task"]
    data_type = param["data_type"]
    preprocessing_steps = param["preprocessing_steps"]

    # =============================================================================================
    # Load the data:
    raw_file = Path(bids_root, "sub-" + subject, "ses-" + session, data_type,
                    "sub-{}_ses-{}_task-{}_{}-raw.fif".format(subject, session, task, data_type))
    raw = mne.io.read_raw_fif(raw_file)
    # Convert the annotations to event for epoching:
    print('Creating annotations')
    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # =============================================================================================
    # Loop through the preprocessing steps:
    for step in preprocessing_steps:
        # Extract the parameters of the current step:
        step_param = param[step]

        # Performing blinks, saccades and fixaction extraction:
        if step in ["extract_blinks", "extract_saccades", "extract_fixation"]:
            print("Extracting the {} from the annotation".format(step_param["description"]))
            raw = extract_eyelink_events(raw, step_param["description"])

        if step == "epochs":
            # Looping through each of the different epochs file to create:
            for epoch_name in step_param.keys():
                epochs = epoch_data(raw, events_from_annot, event_dict, **step_param[epoch_name])
                # Save this epoch to file:
                save_root = Path(bids_root, "derivatives", "preprocessing", "sub-" + subject,
                                 "ses-" + session, data_type)
                if not os.path.isdir(save_root):
                    os.makedirs(save_root)
                # Generate the file name:
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(subject, session, task, data_type,
                                                                              epoch_name)
                # Save:
                epochs.save(Path(save_root, file_name), overwrite=True)

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


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX110", "SX111"]
    parameters_file = r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker" \
                      r"\parameters\preprocessing_parameters.json "
    for sub in subjects_list:
        preprocessing(sub, parameters_file)
