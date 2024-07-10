import os
import json
import pickle
import mne
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator
import environment_variables as ev
from helper_function.helper_general import create_super_subject, get_roi_channels, \
    decoding, moving_average

# Set list of views:
views = ['lateral', 'medial', 'rostral', 'caudal', 'ventral', 'dorsal']


def decoding_pipeline(parameters_file, subjects, data_root, analysis_name="decoding", task_conditions=None):
    """
    Perform decoding analysis on iEEG data.

    :param parameters_file: (str) Path to the parameters file in JSON format.
    :param subjects: (list) List of subjects.
    :param data_root: (str) Root directory for data.
    :param analysis_name: (str) Name of the analysis.
    :param task_conditions: (list) List of task conditions.
    :return: None
    """
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--roi', type=str, default=None,
                        help="Name of the ROI on which to run the analysis")
    args = parser.parse_args()

    if task_conditions is None:
        task_conditions = {"tr": "Relevant non-target", "ti": "Irrelevant", "targets": "Target"}

    with open(args.config) as json_file:
        param = json.load(json_file)

    save_dir = Path(ev.bids_root, "derivatives", analysis_name, param["task"], args.config.split(".json")[0])
    os.makedirs(save_dir, exist_ok=True)

    roi_results = {}
    times = []

    if len(args.roi) == 0:
        labels = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', hemi='both', surf_name='pial',
                                            subjects_dir=ev.fs_directory, sort=True)
        roi_names = list(set(lbl.name.replace("-lh", "").replace("-rh", "")
                             for lbl in labels if "unknown" not in lbl.name))
    elif isinstance(args.roi, str):
        roi_names = [args.roi]
    else:
        raise Exception("The ROI must be a single string!")

    for ii, roi in enumerate(roi_names):
        print("=========================================")
        print("ROI")
        print(roi)
        subjects_epochs = {}
        roi_results[roi] = {}

        # Convert to volumetric labels:
        vol_roi = ["ctx_lh_" + roi, "ctx_rh_" + roi]
        print(vol_roi)

        for sub in subjects:
            epochs_file = Path(data_root, "derivatives", "preprocessing", f"sub-{sub}", f"ses-{param["session"]}",
                               param["data_type"], param["preprocessing_folder"], param["signal"],
                               f"sub-{sub}_ses-{param["session"]}_task-{param["task"]}_desc-epoching_ieeg-epo.fif")
            epochs = mne.read_epochs(epochs_file)

            # Extract the conditions of interest:
            epochs = epochs[param["conditions"]]

            # Crop the data:
            epochs.crop(tmin=param["crop"][0], tmax=param["crop"][1])

            # Get the channels within this ROI:
            picks = get_roi_channels(data_root, sub, param["session"], param["atlas"], vol_roi)
            # Skip if no channels in that ROI for this subject
            if not picks:
                print(f"sub-{sub} has no electrodes in {roi}")
                continue
            # Append to the rest:
            subjects_epochs[sub] = epochs.pick(picks)

        # Skip if no channels were found in this ROI:
        if not subjects_epochs:
            continue

        # 1. Create the model:
        # ============================
        clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', class_weight='balanced'))
        time_res = SlidingEstimator(clf, n_jobs=1, scoring=param["metric"], verbose="ERROR")

        # Loop through each task conditions:
        for tsk in task_conditions.keys():
            # 2. Create the super subject:
            # ============================
            tsk_epochs = {sub: subjects_epochs[sub][task_conditions[tsk]] for sub in subjects_epochs}
            times = [tsk_epochs[sub].times for sub in tsk_epochs.keys()][0]
            data, labels = create_super_subject(tsk_epochs, param["labels_col"], n_trials=param["min_ntrials"])
            n_channels = data.shape[1]

            # 3. Moving average:
            # ============================
            window_size = int((param["mvavg_window_ms"] * 0.001) / (times[1] - times[0]))
            print(f"Window size: {window_size}")
            data = moving_average(data, window_size, axis=-1, overlapping=False)
            times = moving_average(times, window_size, axis=-1, overlapping=False)

            # 4. Apply decoding (pseudotrials happen inside)
            # ============================
            scores = []
            for i in range(param["n_iter"]):
                # Repeat the decoding sev
                scr = decoding(time_res, data, labels,
                               n_pseudotrials=param["pseudotrials"],
                               kfolds=param["kfold"],
                               verbose=True,
                               label_shuffle=False)
                scores.append(scr)
            # Compute the null distribution:
            scores_shuffle = Parallel(n_jobs=param["n_jobs"])(delayed(decoding)(time_res,
                                                                        data,
                                                                        labels,
                                                                        n_pseudotrials=param["pseudotrials"],
                                                                        kfolds=param["kfold"],
                                                                        verbose=True,
                                                                        label_shuffle=True
                                                                        ) for i in
                                                tqdm(range(int(param["n_perm"]))))
            # Average across iterations:
            scores = np.mean(np.stack(scores, axis=2), axis=-1)
            scores_shuffle = np.mean(np.stack(scores_shuffle, axis=2), axis=-1)

            # Package the results:
            roi_results[roi].update({
                f"scores_{tsk}": scores,
                f"scores_shuffle_{tsk}": scores_shuffle,
                "n_channels": n_channels,
                "times": times
            })

        # Save results to file:
        with open(Path(save_dir, f'results-{roi}.pkl'), 'wb') as f:
            pickle.dump(roi_results[roi], f)

    # Save results to file:
    with open(Path(save_dir, 'results-all_roi.pkl'), 'wb') as f:
        pickle.dump(roi_results, f)


if __name__ == "__main__":
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\06-iEEG_decoding_parameters_all-dur.json"
    )
    bids_root = "/hpc/users/alexander.lepauvre/ReconstructedTime/bids-curate"
    decoding_pipeline(parameters, ev.subjects_lists_ecog["dur"], bids_root,
                      analysis_name="decoding",
                      task_conditions={"tr": "Relevant non-target", "ti": "Irrelevant"})
