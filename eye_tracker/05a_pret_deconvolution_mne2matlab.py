import numpy as np
from scipy.io import savemat
import environment_variables as ev
from pathlib import Path
import os
import mne


def epochs2mat(subjects, output_dir, session="1", task="prp", conditions_filter=None, factors=None,
               rt_column="RT_aud", decim_factor=5):
    """

    :param subjects:
    :param output_dir:
    :param session:
    :param task:
    :param conditions_filter:
    :param factors:
    :param rt_column:
    :param decim_factor:
    :return:
    """
    # Create the output directory:
    if factors is None:
        factors = ["SOA", "duration", "lock"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Loop through each subject:
    for subject in subjects:
        # Load the data:
        epochs = mne.read_epochs(Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + subject,
                                      "ses-" + session, "eyetrack",
                                      "sub-{}_ses-{}_task-{}_eyetrack_desc-visual_onset-epo.fif".format(subject,
                                                                                                        session,
                                                                                                        task)))
        if conditions_filter is not None:
            epochs = epochs[conditions_filter]
        # Downsample the data:
        if decim_factor > 0:
            epochs.decimate(decim_factor)

        # ====================================================================================
        # Format the metadata to work well with matlab format down the line:
        # Prepare the conditions vector:
        if len(factors) != 0:
            metadata = epochs.metadata[factors + [rt_column]].copy().reset_index(drop=True)
            # Sort the conditions to ensure that the string encoding produces the same labels:
            metadata_sorted = metadata.sort_values(factors).replace("-", "", regex=True)
            trials_order = list(metadata_sorted.index)
            # Replace the floats by strings:
            metadata_strings = metadata_sorted.copy()[factors]
            for col in metadata_strings.columns:
                try:
                    float(metadata_strings.loc[0, col])
                    unique_vals = metadata_strings[col].unique()
                    encoding_dict = {value: f"{col}{i + 1}" for i, value in enumerate(unique_vals)}
                    metadata_strings[col] = metadata_strings[col].map(encoding_dict)
                except ValueError:
                    continue
            cond_labels_np = metadata_strings.to_numpy()
            cond_labels_joined = np.array(['_'.join(cond_labels_np[i, :]) for i in range(cond_labels_np.shape[0])])
        else:
            trials_order = list(range(epochs.metadata.shape[0]))
            cond_labels_joined = ["cond_1"] * epochs.metadata.shape[0]

        # ====================================================================================
        # Create the latency matrix:
        if task == "prp":
            latency_matrix = []
            for i, trial in metadata_sorted.iterrows():
                lats = np.zeros((3, 1))
                if trial["lock"] == "offset":
                    lats[1] = float(trial["SOA"]) + float(trial["duration"])
                else:
                    lats[1] = float(trial["SOA"])
                lats[2] = float(trial["duration"])
                latency_matrix.append(lats)
            # Convert to a dataframe:
            latency_matrix = np.squeeze(np.array(latency_matrix))
            events = np.array(['visOnset', 'audOnset', 'visOffset'])

            # Create the box_times matrix. We assume two boxes for the trials, one for T1 and one for T2. As T1 does not
            # require responses on some trials, we assume the mean RT of when response is required.
            box_times_matrix = []
            for i, trial in metadata_sorted.iterrows():
                trial_box_times = np.zeros((4, 1))
                # The onset of T1 is always 0:
                trial_box_times[0] = 0
                # The end of the box is the median reaction time of the trials in which there was a response:
                trial_box_times[1] = 0.900
                # For T2, the onset is SOA:
                if trial["lock"] == "offset":
                    trial_box_times[2] = float(trial["SOA"]) + float(trial["duration"])
                else:
                    trial_box_times[2] = float(trial["SOA"])
                # And the end of the box is the RT. Add to the RT the latency of the audio signal:
                trial_box_times[3] = trial_box_times[2] + trial[rt_column]
                box_times_matrix.append(trial_box_times)
            box_times_matrix = np.squeeze(np.array(box_times_matrix))
            box_labels = np.array(['T1-related', 'T2-related'])
        elif task == "auditory":
            trials_order = list(range(epochs.metadata.shape[0]))
            latency_matrix = np.zeros((epochs.metadata.shape[0], 1))
            events = np.array(['audOnset'])
            # Create the box_times matrix.
            box_times_matrix = np.zeros((latency_matrix.shape[0], 2))
            box_times_matrix[:, 1] = epochs.metadata[rt_column]
            box_times_matrix = np.squeeze(np.array(box_times_matrix))
            box_labels = np.array(['T1-related'])
        elif task == "visual":
            latency_matrix = np.zeros((epochs.metadata.shape[0], 2))
            latency_matrix[:, 1] = metadata_sorted["duration"].astype(float)
            # Convert to a dataframe:
            latency_matrix = np.squeeze(np.array(latency_matrix))
            events = np.array(['visOnset', 'visOffset'])
            # Create the box_times matrix.
            box_times_matrix = np.zeros((epochs.metadata.shape[0], 2))
            box_times_matrix[:, 1] = 0.900
            box_times_matrix = np.squeeze(np.array(box_times_matrix))
            box_labels = np.array(['T1-related'])

        # Save to a .mat file
        savemat(Path(output_dir, "sub-{}_ses-{}_task-{}_epochs.mat".format(subject, session, task)),
                {'data': np.mean(epochs.get_data(copy=False, picks=["pupil_left", "pupil_right"]), axis=1)
                [trials_order, :],
                 'tmin': epochs.times[0],
                 'tmax': epochs.times[-1],
                 'sfreq': epochs.info["sfreq"],
                 'latency': latency_matrix,
                 'eventlabels': events,
                 'cond_labels': cond_labels_joined,
                 'box_times': box_times_matrix,
                 'box_labels': box_labels
                 }
                )


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX119", "SX120", "SX121"]
    tasks_list = ["auditory", "visual", "prp"]
    factors = [[], ["duration"], ["SOA", "duration", "lock", "task relevance"]]
    for i, tsk in enumerate(tasks_list):
        epochs2mat(subjects_list, Path(ev.bids_root, "derivatives", "pret"), session="1", task=tsk,
                   conditions_filter=["non-target", "irrelevant"],
                   factors=factors[i], rt_column="RT_aud", decim_factor=5)
