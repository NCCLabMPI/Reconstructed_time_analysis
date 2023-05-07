import os
import mne
import json
from pathlib import Path
import pandas as pd
import numpy as np
from eye_tracker.preprocessing_helper_function import extract_blink, epoch_data


def preprocessing(subject, parameters):
    """
    This function preprocesses the eyetracking data, using several MNE key functionalities for handling the data
    :param subject:
    :param parameters:
    :return:
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
            raw = extract_blink(raw, step_param["description"])

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
                epochs.save(Path(save_root, file_name))


if __name__ == "__main__":
    sub = "SX105"
    parameters_file = r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker" \
                      r"\parameters\preprocessing_parameters.json "
    preprocessing(sub, parameters_file)
