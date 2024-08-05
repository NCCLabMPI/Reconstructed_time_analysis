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
    roi_cts = {}

    for ii, roi in enumerate(roi_names):
        # print("=========================================")
        # print("ROI")
        # print(roi)
        subjects_epochs = {}
        roi_results[roi] = {}
        roi_cts[roi] = 0

        # Convert to volumetric labels:
        vol_roi = ["ctx_lh_" + roi, "ctx_rh_" + roi]
        # print(vol_roi)

        for sub in subjects:
            # Get the channels within this ROI:
            picks = get_roi_channels(data_root, sub, param["session"], param["atlas"], vol_roi)
            roi_cts[roi] += len(picks)
        print(f'{roi}, {roi_cts[roi]}')


if __name__ == "__main__":
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\06-iEEG_decoding_parameters_all-dur.json"
    )
    bids_root = r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-curate"
    decoding_pipeline(parameters, ev.subjects_lists_ecog["dur"], bids_root,
                      analysis_name="decoding",
                      task_conditions={"tr": "Relevant non-target", "ti": "Irrelevant"})
