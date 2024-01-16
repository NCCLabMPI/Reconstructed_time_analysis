import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling
import seaborn as sns
import pandas as pd
import os
import environment_variables as ev

# Set the font size:
plt.rcParams.update({'font.size': 22})


def fixation_analysis(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    fixation_proportion = pd.DataFrame()
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
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])

        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract the data and average across left and right eye:
                        data = np.mean(epochs.copy()["/".join([task_rel, duration, lock, soa])].get_data(copy=False),
                                       axis=1)
                        # Threshold the data according to the fixation distance threshold:
                        data_thresh = data < param["fixdist_thresh_deg"]
                        # Average across trials:
                        data_trial_avg = np.mean(data_thresh, axis=0)
                        # Average time:
                        fix_prop = np.mean(data_trial_avg, axis=0)

                        # Add to data frame using pd.concat:
                        fixation_proportion = pd.concat([fixation_proportion,
                                                         pd.DataFrame({"subject": sub,
                                                                       "task_relevance": task_rel,
                                                                       "duration": duration,
                                                                       "lock": lock,
                                                                       "soa": soa,
                                                                       "fixation_proportion": fix_prop},
                                                                      index=[0])])
    fixation_proportion = fixation_proportion.reset_index(drop=True)
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "fixation", "group_level", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    fixation_proportion.to_csv(Path(save_dir, "fixation_proportion.csv"))


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\02-fixation_proportion_parameters.json ")
    fixation_analysis(parameters, subjects_list)
