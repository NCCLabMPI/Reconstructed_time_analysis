import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helper_function.helper_general import load_beh_data
import seaborn as sns
from helper_function.helper_plotter import plot_pupil_latency, soa_boxplot
import environment_variables as ev
import pandas as pd

# Set the font size:
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def pupil_beh_correlation(parameters_file, session="1", experiment="prp", analysis_name="pupil_beh_correlation"):
    # Create the directory to load the results;
    data_dir = Path(ev.bids_root, "derivatives", "pupil_latency", experiment)

    # Create the directory to store the results:
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load the pupil latency data:
    pupil_latency = pd.read_csv(Path(data_dir, "pupil_peak_latencies.csv"), index_col=False)

    # Load the behavioral data of each subject:
    file_name = "sub-{}_ses-{}_run-all_task-{}_events.csv"
    if isinstance(session, list):
        beh_df = []
        for ses in session:
            beh_df.append(load_beh_data(ev.bids_root, ev.subjects_lists_beh[experiment], file_name, session=ses,
                                        task=experiment, do_trial_exclusion=True))
        beh_df = pd.concat(beh_df).reset_index(drop=True)
    else:
        beh_df = load_beh_data(ev.bids_root, ev.subjects_lists_beh[experiment], file_name, session=session,
                               task=experiment, do_trial_exclusion=True)
    # Average PRP per SOA, onset offset and task relevance conditions and subjects:
    beh_avg = beh_df.groupby(['sub_id', 'SOA', 'task_relevance', 'duration', 'SOA_lock']).agg(
        {'RT_aud': 'mean'}).reset_index()
    beh_avg = beh_avg.sort_values(by=['sub_id', 'SOA', 'task_relevance', 'duration', 'SOA_lock']).reset_index(
        drop=True)
    pupil_latency = pupil_latency.sort_values(by=['sub_id', 'SOA', 'task', 'duration', 'SOA_lock']).reset_index(
        drop=True)[['sub_id', 'SOA', 'task', 'duration', 'SOA_lock', 'latency_aud']]

    # Remove subjects for which we don't have eyetracking data:
    beh_avg = beh_avg[beh_avg["sub_id"].isin(np.unique(pupil_latency["sub_id"].to_numpy()))]

    # Rename the columns for merge:
    beh_avg = beh_avg.rename(columns={"task_relevance": "task"})
    beh_avg = beh_avg.rename(columns={"RT_aud": "RT2"})
    pupil_latency = pupil_latency.rename(columns={"latency_aud": "pupil_latency"})
    # Merge the two:
    beh_et_tbl = pd.merge(pupil_latency, beh_avg)
    # Z score the data:
    beh_et_tbl = beh_et_tbl.assign(
        RT2_z=beh_et_tbl.groupby('sub_id')['RT2'].transform(lambda x: (x - x.mean()) / x.std()),
        PupilLatency_z=beh_et_tbl.groupby('sub_id')['pupil_latency'].transform(lambda x: (x - x.mean()) / x.std())
    )
    # Save to file:
    beh_et_tbl.to_csv(Path(save_dir, "pupil_RT2_correlation.csv"))

    # Remove the targets:
    beh_et_tbl = beh_et_tbl[beh_et_tbl["task"] != "target"]
    # Remove offset trials:
    beh_et_tbl = beh_et_tbl[beh_et_tbl["SOA_lock"] != "offset"]

    # Plot the correlation:
    plt.figure(figsize=(10, 6))
    sns.lmplot(x="PupilLatency_z", y="RT2_z", hue="sub_id",
               data=beh_et_tbl, line_kws={'linewidth': 0.8, "alpha": 0.6}, legend=False, palette='tab20')
    # Plot the correlation across all participants:
    sns.regplot(x="PupilLatency_z", y="RT2_z", data=beh_et_tbl, scatter=False,
                line_kws={'color': 'black', 'linewidth': 2})
    plt.savefig(Path(save_dir, f"pupil_RT2_correlation.svg"))
    plt.close()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\05-ET_pupil_latency_parameters.json")

    # ==================================================================================
    # PRP analysis:
    pupil_beh_correlation(parameters, experiment="prp", session="1",
                          analysis_name="pupil_beh_correlation")

    # ==================================================================================
    # Introspection analysis:
    pupil_beh_correlation(parameters, experiment="introspection",
                          session=["2", "3"], analysis_name="pupil_beh_correlation")
