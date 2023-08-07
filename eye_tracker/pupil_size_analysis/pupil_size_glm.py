import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eye_tracker.general_helper_function import baseline_scaling
import os
import statsmodels.api as sm


# First, load the parameters:
bids_root = r"C:\\Users\\alexander.lepauvre\\Documents\\PhD\\Reconstructed_Time\\bids"
visit = "1"
task = "prp"
session = "1"
data_type = "eyetrack"
epoch_name = "visual_onset"
crop = [-0.3, 2.0]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
regressors = ["SOA"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
locks = ["onset", "offset"]
subjects = ["SX102", "SX105", "SX106", "SX107", "SX108", "SX110", "SX111"]

# Preallocate the subjects betas:
subjects_betas = []

# Loop through each subject:
for sub in subjects:
    # Load the epochs:
    root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                data_type)
    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, task, data_type,
                                                                  epoch_name)
    epochs = mne.read_epochs(Path(root, file_name))
    # Crop if needed:
    epochs.crop(crop[0], crop[1])
    # Extract the relevant channels:
    epochs.pick(picks)
    # Baseline correction:
    baseline_scaling(epochs, correction_method="mean", baseline=(None, -0.05))
    # Extract the regressors:
    X = epochs.metadata[[regressors]].to_list()
    x_ols = sm.add_constant(X)
    # Extract the data:
    data = epochs.get_data()
    # Average across both eyes:
    data = np.nanmean(data, axis=1)
    betas = []
    # Loop through each time point:
    for t in range(data.shape[-1]):
        # Extract the data of this time point:
        y = np.squeeze(data[..., t])
        # Compute the regressions:
        result = sm.OLS(x_ols, y).fit()
        # Store in array:
        betas.append(result.params)
    # Convert to an array:
    subjects_betas.append(np.array(betas))

# Plot the outcome:

# Compute cluster based permutation test:

