import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helper_function.helper_general import (baseline_scaling, max_percentage_index, equate_epochs_events,
                                            reject_bad_epochs, format_drop_logs)
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


def pupil_latency(parameters_file, subjects, session="1", experiment="prp", analysis_name="pupil_latency",
                  reject_bad_trials=True):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Load all subjects data:
    subjects_epochs = {sub: None for sub in subjects}
    # Create the directory to save the results in:
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # The soas differ between the experiments, therefore, extracting them directly from the epochs objects:
    soas = []
    # Loop through each subject:
    for sub in subjects:
        print("Loading sub-{}".format(sub))
        if isinstance(session, list):
            epochs = []
            for ses in session:
                root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + ses,
                            param["data_type"])
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, ses, experiment,
                                                                           param["data_type"])
                epochs.append(mne.read_epochs(Path(root, file_name)))
            # Equate the epochs events.
            epochs = equate_epochs_events(epochs)
            epochs = mne.concatenate_epochs(epochs, add_offset=True)
        else:
            root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                        param["data_type"])
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, experiment,
                                                                       param["data_type"])
            epochs = mne.read_epochs(Path(root, file_name))
        # Extract the soas:
        soas = list(epochs.metadata["SOA"].unique())
        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        # Reject bad epochs according to predefined criterion:
        if reject_bad_trials:
            reject_bad_epochs(epochs,
                              baseline_window=param["baseline_window"],
                              z_thresh=param["baseline_zthresh"],
                              eyes=param["eyes"],
                              exlude_beh=param["exlude_beh"],
                              remove_blinks=param["remove_blinks"],
                              blinks_window=param["blinks_window"],
                              events_bound_blinks=param["events_bound_blinks"],
                              remove_fixdist=param["remove_fixdist"],
                              fixdist_thresh_deg=param["fixdist_thresh_deg"],
                              fixdist_prop_trhesh=param["fixdist_prop_trhesh"])
        # Extract the relevant channels:
        epochs.pick(param["picks"])
        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])
        subjects_epochs[sub] = epochs

    # Extract the times of the first subject (assuming that it is the same for all subjects, which it should be):
    times = subjects_epochs[subjects[0]].times

    # Plot the drop logs:
    drop_log_df = format_drop_logs({sub: subjects_epochs[sub].drop_log for sub in subjects_epochs.keys()})
    # Plot the drop log:
    # Extract the columns:
    cols = [col for col in drop_log_df.columns if col != "sub"]
    fig, ax = plt.subplots(figsize=[8.3, 8.3])
    ax.boxplot([drop_log_df[col].to_numpy() for col in cols], labels=cols)
    ax.axhline(param["drop_trials_threshold"], linestyle="--", color="r")
    ax.set_ylabel("Proportion dropped trials")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(Path(save_dir, "drop_log.svg"), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "drop_log.png"), transparent=True, dpi=300)
    plt.close()

    # Extract the subject that exceed the proportion of dropped trials
    drop_subjects = drop_log_df.loc[drop_log_df["total"] >= param["drop_trials_threshold"], "sub"].to_list()
    for sub in drop_subjects:
        del subjects_epochs[sub]

    # ==================================================================================================================
    # Create LMM tables:
    # ===========================================================
    latencies_lmm = []
    for soa in soas:
        for task in param["task_relevance"]:
            for duration in param["duration"]:
                for lock in param["lock"]:
                    for sub in subjects_epochs.keys():
                        if lock == "offset":
                            onset_soa = float(soa) + float(duration)
                        else:
                            onset_soa = float(soa)
                        # Average the data across both eyes:
                        if lock == "offset":
                            # Extract the data of the particular condition and crop the data:
                            sub_epochs = subjects_epochs[sub].copy().crop(float(duration) + float(soa),
                                                                          subjects_epochs[sub].times[-1])[
                                "/".join([soa, task, duration, lock])]
                        else:
                            # Extract the data of the particular condition:
                            sub_epochs = subjects_epochs[sub].copy().crop(float(soa),
                                                                          subjects_epochs[sub].times[-1])[
                                "/".join([soa, task, duration, lock])]
                        try:
                            data = np.nanmean(sub_epochs, axis=1)
                        except np.exceptions.AxisError:
                            latencies_lmm.append(pd.DataFrame({
                                "sub_id": sub,
                                "SOA": float(soa),
                                "task": task,
                                "duration": float(duration),
                                "SOA_lock": lock,
                                "onset_SOA": onset_soa,
                                "latency": np.nan,
                                "latency_aud": np.nan,
                                "amplitude": np.nan
                            }, index=[0]))
                            continue

                        # Average across trials
                        evk = np.mean(data, axis=0)
                        ind, val = max_percentage_index(evk, 90)
                        latency = sub_epochs.times[ind]
                        latency_aud = sub_epochs.times[ind] - onset_soa

                        latencies_lmm.append(pd.DataFrame({
                            "sub_id": sub,
                            "SOA": float(soa),
                            "task": task,
                            "duration": float(duration),
                            "SOA_lock": lock,
                            "onset_SOA": onset_soa,
                            "latency": latency,
                            "latency_aud": latency_aud,
                            "amplitude": val
                        }, index=[0]))
    # Convert to data frame:
    latencies_lmm = pd.concat(latencies_lmm).reset_index(drop=True)
    # Save the peak latencies:
    latencies_lmm.to_csv(Path(save_dir, "pupil_peak_latencies.csv"))

    # ==================================================================================================================
    # Plotting latencies:
    # ===============================================================================================
    colors_onset = [ev.colors["soa_onset_locked"][str(soa)] for soa in np.sort(latencies_lmm["SOA"].unique())]
    colors_offset = [ev.colors["soa_offset_locked"][str(soa)] for soa in np.sort(latencies_lmm["SOA"].unique())]
    # Target
    fig_ta, ax_ta = soa_boxplot(latencies_lmm[latencies_lmm["task"] == 'target'],
                                "latency_aud",
                                colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                                fig_size=[8.3 / 3, 11.7 / 3], label=r"$\tau$")

    # Task relevant:
    fig_tr, ax_tr = soa_boxplot(latencies_lmm[latencies_lmm["task"] == 'non-target'],
                                "latency_aud",
                                colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                                fig_size=[8.3 / 3, 11.7 / 3], label=r"$\tau$")
    # Task relevant:
    fig_ti, ax_ti = soa_boxplot(latencies_lmm[latencies_lmm["task"] == 'irrelevant'],
                                "latency_aud",
                                colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                                fig_size=[8.3 / 3, 11.7 / 3], label=r"$\tau$")
    lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
    max_lims = [min(min(lims)), max(max(lims))]
    ax_tr[0].set_ylim(max_lims)
    ax_ti[0].set_ylim(max_lims)
    ax_ti[-1].legend()
    fig_ta.suptitle("Target")
    fig_tr.suptitle("Relevant non-target")
    fig_ti.suptitle("Irrelevant non-target")
    fig_ta.text(0.5, 0, 'Time (s)', ha='center', va='center')
    fig_tr.text(0.5, 0, 'Time (s)', ha='center', va='center')
    fig_ti.text(0.5, 0, 'Time (s)', ha='center', va='center')
    fig_ta.text(0, 0.9, r'$\tau$ (s)', ha='center', va='center', fontsize=18)
    fig_tr.text(0, 0.9, r'$\tau$ (s)', ha='center', va='center', fontsize=18)
    fig_ti.text(0, 0.9, r'$\tau$ (s)', ha='center', va='center', fontsize=18)
    fig_ta.savefig(Path(save_dir, "pupil_latency_target.svg"), transparent=True, dpi=300)
    fig_ta.savefig(Path(save_dir, "pupil_latency_target.png"), transparent=True, dpi=300)
    fig_tr.savefig(Path(save_dir, "pupil_latency_tr.svg"), transparent=True, dpi=300)
    fig_tr.savefig(Path(save_dir, "pupil_latency_tr.png"), transparent=True, dpi=300)
    fig_ti.savefig(Path(save_dir, "pupil_latency_ti.svg"), transparent=True, dpi=300)
    fig_ti.savefig(Path(save_dir, "pupil_latency_ti.png"), transparent=True, dpi=300)
    plt.close(fig_ta)
    plt.close(fig_tr)
    plt.close(fig_ti)

    # ==================================================================================================================
    # Plot evoked pupil responses:
    # ======================================================================================
    # Plot separately for each duration:
    for task in param["task_relevance"]:
        if experiment == "prp":
            if task == "target":
                pupil_size_ylim = [0.96, 1.25]
            else:
                pupil_size_ylim = [0.96, 1.1]
        else:
            if task == "target":
                pupil_size_ylim = [0.92, 1.25]
            else:
                pupil_size_ylim = [0.92, 1.1]
        # ===========================================================
        # Onset locked:
        latencies = latencies_lmm[(latencies_lmm["task"] == task) & (latencies_lmm["SOA_lock"] == "onset")]
        # Open figure:
        fig, ax = plt.subplots(4, 1, figsize=[8.3 / 3, 11.7 / 3], sharex=True, sharey=True)
        # Plot onset locked:
        soa_evks = {soa: [] for soa in soas}
        for soa in soas:
            # Loop through each subject:
            evks = []
            for sub in subjects_epochs.keys():
                # Average the data across both eyes:
                data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, task, "onset"])], axis=0)
                # Average across both eyes:
                try:
                    evk = np.mean(data, axis=0)
                    evks.append(evk)
                except np.exceptions.AxisError:
                    continue
            soa_evks[soa] = np.mean(np.array(evks), axis=0)
        # Plot each soa:
        for soa in np.sort(list(soa_evks.keys())):
            ax[0].plot(times, soa_evks[soa], color=ev.colors["soa_onset_locked"][soa], label=soa)
            # Plot vline marking the onset of the stimulus:
            ax[0].vlines(x=latencies[latencies["SOA"] == float(soa)]["onset_SOA"].to_numpy()[0],
                         ymin=pupil_size_ylim[0],
                         ymax=pupil_size_ylim[1], linestyle="-",
                         color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
            # Compute the mean latency:
            lat = np.nanmean(latencies[latencies["SOA"] == float(soa)]["latency"].to_numpy())
            amp = soa_evks[soa][np.argmin(np.abs(times - lat))]
            ax[0].vlines(x=lat, ymin=0, ymax=amp, linestyle="--", color=ev.colors["soa_onset_locked"][soa],
                         linewidth=2, zorder=10)

        # ===========================================================
        # Offset locked:
        for dur_i, duration in enumerate(param["duration"]):
            latencies = latencies_lmm[(latencies_lmm["task"] == task) &
                                      (latencies_lmm["SOA_lock"] == "offset") &
                                      (latencies_lmm["duration"] == float(duration))]
            # Plot onset locked:
            soa_evks = {soa: [] for soa in soas}
            for soa in soas:
                # Loop through each subject:
                evks = []
                for sub in subjects_epochs.keys():
                    # Average the data across both eyes:
                    data = np.nanmean(subjects_epochs[sub].copy()["/".join([soa, task, "offset",
                                                                            duration])], axis=0)
                    # Average across both eyes:
                    try:
                        evk = np.mean(data, axis=0)
                        evks.append(evk)
                    except np.exceptions.AxisError:
                        continue
                soa_evks[soa] = np.mean(np.array(evks), axis=0)
            # Plot each soa:
            for soa in np.sort(list(soa_evks.keys())):
                ax[dur_i + 1].plot(times, soa_evks[soa], color=ev.colors["soa_offset_locked"][soa], label=soa)
                # Plot vline marking the onset of the stimulus:
                ax[dur_i + 1].vlines(x=latencies[latencies["SOA"] == float(soa)]["onset_SOA"].to_numpy()[0],
                                     ymin=pupil_size_ylim[0],
                                     ymax=pupil_size_ylim[1], linestyle="-",
                                     color=ev.colors["soa_offset_locked"][soa], linewidth=2, zorder=10)
                # Compute the mean latency and the associated value on the evoked response:
                lat = np.nanmean(latencies[latencies["SOA"] == float(soa)]["latency"].to_numpy())
                amp = soa_evks[soa][np.argmin(np.abs(times - lat))]
                ax[dur_i + 1].vlines(x=lat, ymin=0, ymax=amp, linestyle="--", color=ev.colors["soa_offset_locked"][soa],
                                     linewidth=2, zorder=10)
            # Add a box:
            rectangle = Rectangle((0, pupil_size_ylim[0]), float(duration), pupil_size_ylim[1], linewidth=1,
                                  edgecolor='none', facecolor=[0.8, 0.8, 0.8], alpha=0.5)
            ax[dur_i + 1].add_patch(rectangle)
        # Remove space between plots:
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[-1].set_xlim([times[0], times[-1]])
        ax[-1].set_ylim(pupil_size_ylim)
        ax[-1].set_ylabel("Pupil size (a.u.)")
        ax[-1].set_xlabel("Time from T1 onset (s)")
        ax[0].legend()
        ax[1].legend()
        fig.savefig(Path(save_dir, "pupil_evoked_{}.svg".format(task)), transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "pupil_evoked_{}.png".format(task)), transparent=True, dpi=300)
        plt.close()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\05-ET_pupil_latency_parameters.json")

    # ==================================================================================
    # PRP analysis:
    experiment = "prp"
    pupil_latency(parameters, ev.subjects_lists_et[experiment], experiment=experiment, session="1",
                  analysis_name="pupil_latency", reject_bad_trials=True)
    pupil_latency(parameters, ev.subjects_lists_et[experiment], experiment=experiment, session="1",
                  analysis_name="pupil_latency_no_rej", reject_bad_trials=False)

    # ==================================================================================
    # Introspection analysis:
    experiment = "introspection"
    pupil_latency(parameters, ev.subjects_lists_et[experiment], experiment=experiment, session=["2", "3"],
                  analysis_name="pupil_latency", reject_bad_trials=True)
    pupil_latency(parameters, ev.subjects_lists_et[experiment], experiment=experiment, session=["2", "3"],
                  analysis_name="pupil_latency_no_rej", reject_bad_trials=False)
