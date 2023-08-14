import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eye_tracker.general_helper_function import baseline_scaling
import statsmodels.api as sm

prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']
# First, load the parameters:
bids_root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\bids"
visit = "1"
task = "prp"
session = "1"
data_type = "eyetrack"
epoch_name = "visual_onset"
crop = [-0.3, 3.5]
audio_lock_window = [0., 2.0]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
regressors = ["task relevance", "duration", "SOA"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
locks = ["onset", "offset"]
subjects = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX110", "SX113"]

# Preallocate the subjects betas:
subjects_betas = []

# Loop through each subject:
for sub in subjects:
    print(sub)
    # ===========================================
    # Data loading:
    # Load the epochs:
    root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                data_type)
    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, task, data_type,
                                                                  epoch_name)
    epochs = mne.read_epochs(Path(root, file_name))
    # Extract the relevant conditions:
    epochs = epochs[["non-target/onset", "irrelevant/onset"]]
    # Crop if needed:
    epochs.crop(crop[0], crop[1])
    # epochs.decimate(10)
    # Extract the relevant channels:
    epochs.pick(picks)

    # ===========================================
    # Preprocessing
    # Baseline correction:
    baseline_scaling(epochs, correction_method="mean", baseline=(None, -0.05))
    #  Extract the metadata:
    meta_data = epochs.metadata
    # Convert the task relevance to a dummy variable:
    task_regressor = meta_data["task relevance"].map({"non-target": -1, "irrelevant": 1}).to_list()
    soa_regressor = meta_data["SOA"].astype(float).tolist()
    X = np.array([task_regressor, soa_regressor])
    x_ols = sm.add_constant(X.T)

    # Extract the data time locked to the auditory stimulus onset:
    data = epochs.get_data(["LPupil", "RPupil"])
    data_audio_locked = []
    audio_lock_nsamples = int(audio_lock_window[1] * epochs.info["sfreq"])
    for trial in range(data.shape[0]):
        # Get the SOA:
        soa = meta_data["SOA"].iloc[trial]
        # Find the sample the SOA corresponds to:
        soa_sample = np.where(epochs.times >= float(soa))[0][0]
        # Extract the data:
        data_audio_locked.append(data[trial, :, soa_sample:soa_sample + audio_lock_nsamples])
    # Convert to an array:
    data_audio_locked = np.array(data_audio_locked)
    # Average across both eyes:
    data = np.nanmean(data_audio_locked, axis=1)
    betas = []
    # Loop through each time point:
    for t in range(data.shape[-1]):
        # Extract the data of this time point:
        y = np.squeeze(data[..., t])
        # Compute the regressions:
        result = sm.OLS(x_ols, y).fit()
        # print(f"coefficient of determination: {result.rsquared}")
        # Store in array:
        betas.append(result.params)
    # Plot the data and the betas:
    n_soas = len(np.unique(soa_regressor))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    for ind, soa in enumerate(np.unique(soa_regressor)):
        # Compute the mean and error of this SOA:
        avg = np.mean(np.squeeze(data[np.where(soa_regressor == soa), :]), axis=0)
        err = sem(np.squeeze(data[np.where(soa_regressor == soa), :]), axis=0)
        ax[0].plot(np.linspace(audio_lock_window[0], audio_lock_window[1], data.shape[-1]),
                            avg, label=soa)
        #
        ax[0].fill_between(np.linspace(audio_lock_window[0], audio_lock_window[1], data.shape[-1]),
                                    avg - err, avg + err,
                                    alpha=0.3)
    ax[0].set_ylabel("Pupil size (norm.)")
    ax[0].legend()
    ax[1].plot(np.linspace(audio_lock_window[0], audio_lock_window[1], data.shape[-1]),
                        np.squeeze(np.array(betas))[:, 1:3], label=["Task", "SOA"])
    ax[1].set_ylabel("Beta (pupil size norm.)")
    ax[1].set_xlabel("Time (s)")
    plt.legend()
    plt.show()
    # Convert to an array:
    subjects_betas.append(np.squeeze(np.array(betas)))

# Plot the average across subjects:

plt.plot(np.linspace(audio_lock_window[0], audio_lock_window[1], data.shape[-1]),
         np.nanmean(np.array(subjects_betas), axis=0)[:, 2], label=["SOA"])
# Add the confidence interval:
# plt.fill_between(np.linspace(audio_lock_window[0], audio_lock_window[1], data.shape[-1]), np.nanmean(np.array(subjects_betas), axis=0)[:, 2] - sem(np.array(subjects_betas)[:, 2],
#                                                                                           axis=0),
#                     np.nanmean(np.array(subjects_betas), axis=0)[:, 2] + sem(np.array(subjects_betas)[:, 2],
#                                                                                axis=0), alpha=0.2)
plt.legend()
plt.show()
# Compute cluster based permutation test:
