import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np
from eye_tracker.general_helper_function import get_event_ts
from eye_tracker.general_helper_function import cousineau_morey_correction
import environment_variables as ev
from mne.stats import bootstrap_confidence_interval

font = {'size': 12}
matplotlib.rc('font', **font)
# Get matplotlib colors:
prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']


def latency_raster_plot(epochs, lock, durations, soas, channels=None, audio_lock=True):
    """
    This function generate events plots from mne epochs object containing eyetracking events data. It loops through
    each durations and SOAs conditions (from our PRP experiment) and generates one subplot per duration. Each subplot is
    actually 2 separate subplots. The first is a raster plot that depicts the time stamp of each event in a trial. The
    second is a boxplot showing the latency of the response per soa.
    :param epochs: (mne epochs object) contains the eyetracking event data. The events data are time series of zeros
    and ones, ones marking when the event of interest occured (saccades, blinks, fixation).
    :param lock: (string) whether the soa event were locked to the onset or to the offset of the stimuli.
    :param durations: (list of strings) list of the duration condition.
    :param soas: (list of strings) list of the soa codnitions
    :param channels: (list of strings) name of the channels containing the event of interest. They come in pairs, right
    and left eye.
    :param audio_lock: (bool) the data were epochjed relative to the visual stimulus onset. But we can easily convert
    to locked the auditory stimulus by cropping the data from the SOA duration onwards.
    :return: fig: the plotted figure
    """
    if channels is None:
        channels = ["Lblink", "Rblink"]

    # Open a figure:
    fig, ax = plt.subplots(figsize=[11, 9], sharex=True)
    gs = GridSpec(len(durations) * 3, 1)

    # Plot separately for the different durations:
    for dur_ind, dur in enumerate(durations):
        data = []
        trials_soas = []
        colors = []
        for ind, soa in enumerate(soas):
            # Extract the data:
            if audio_lock:  # To have the data locked to the audio, crop at SOA
                if lock == 'onset':  # If onset locked, crop from SOA to SOA + 1sec
                    epochs_cropped = epochs.copy()["/".join([lock, dur, soa])].crop(float(soa),
                                                                                    float(soa) + 1)
                else:  # If offset locked, we need to add the trial duration on top
                    epochs_cropped = epochs.copy()["/".join([lock, dur, soa])].crop(float(soa) +
                                                                                    float(dur),
                                                                                    float(soa) +
                                                                                    float(dur) + 1)
                data_l = np.squeeze(epochs_cropped.get_data(picks=channels[0]))
                data_r = np.squeeze(epochs_cropped.get_data(picks=channels[1]))

            else:
                data_l = np.squeeze(epochs.copy()["/".join([lock, dur, soa])].get_data(picks=channels[0]))
                data_r = np.squeeze(epochs.copy()["/".join([lock, dur, soa])].get_data(picks=channels[1]))

            # Combine both eyes such that if we have a 1 in both arrays, then we have a 1, otherwise a 0:
            data.append(np.logical_and(data_l, data_r).astype(int))
            # Colors:
            colors.append([cwheel[ind]] * data_l.shape[0])
            # Append the trial SOAs:
            trials_soas.extend([soa] * data_l.shape[0])
        if audio_lock:
            times = np.arange(0, 1 + 1 / epochs.info['sfreq'], 1 / epochs.info['sfreq'])
        else:
            times = epochs.times

        # Extract the time stamp of each particular event:
        onset_ts = get_event_ts(np.concatenate(data), times)

        # Create the event plot:
        ax_event = plt.subplot(gs[dur_ind * 3:dur_ind * 3 + 2])
        ax_event.eventplot(onset_ts, colors=np.concatenate(colors), linelengths=8, linewidths=1.5,
                           lineoffsets=np.arange(0, len(onset_ts)))
        ax_event.set_ylim([0, len(onset_ts)])
        ax_event.set_xlim([times[0], times[-1]])
        plt.setp(ax_event.get_xticklabels(), visible=False)

        # Create the boxplot underneath it:
        # Reformat the data:
        onset_ts = [[onset_ts[i] for i in list(np.where(np.array(trials_soas) == soa)[0])] for soa in soas]
        onset_ts = [[item for sublist in l for item in sublist] for l in onset_ts]
        # Create the boxplot:
        ax_boxplot = plt.subplot(gs[dur_ind * 3 + 2])
        bplot = ax_boxplot.boxplot(onset_ts, vert=False, patch_artist=True,  # fill with color
                                   labels=soas, whis=[5, 95], showfliers=False)
        # Set face colors:
        for patch, color in zip(bplot['boxes'], cwheel[0:len(soas)]):
            patch.set_facecolor(color)
        ax_boxplot.set_xlim([times[0], times[-1]])

        # Finally, if we are not plotting the data time locked the the audio stim, displaying the stimulus duration
        # as a gray patch:
        if not audio_lock:
            rect = patches.Rectangle((0, 0), float(dur), ax_boxplot.get_ylim()[1],
                                     linewidth=1, edgecolor='none', facecolor=[0.5, 0.5, 0.5], alpha=0.1,
                                     zorder=1)
            ax_boxplot.add_patch(rect)
            rect = patches.Rectangle((0, 0), float(dur), ax_event.get_ylim()[1],
                                     linewidth=1, edgecolor='none', facecolor=[0.5, 0.5, 0.5], alpha=0.1,
                                     zorder=1)
            ax_event.add_patch(rect)
        # Axes deco:
        if dur_ind == 0:
            ax_event.set_title("Lock: {}".format(lock))
        if dur_ind != len(durations) - 1:
            plt.setp(ax_boxplot.get_xticklabels(), visible=False)
        else:
            ax_boxplot.set_ylabel('SOA')
            ax_boxplot.set_xlabel('Time (sec)')
            ax_event.set_ylabel('Trial')
        ax_event.spines[['right', 'top']].set_visible(False)
        ax_boxplot.spines[['right', 'top']].set_visible(False)

    return fig


def plot_within_subject_boxplot(data_df, within_column, between_column, dependent_variable, positions=None,
                                ax=None, cousineau_correction=True, title="", xlabel="", ylabel="", xlim=None,
                                width=0.1, face_colors=None, edge_colors=None, xlabel_fontsize=9):
    """
    This function generates within subject design boxplot with line plots connecting each subjects dots across
    conditions. Further offers the option to apply Cousineau Morey correction. Importantly, data must be passed before
    averaging. The within subject averaging occurs in the function directly!
    :param data_df: (data frame) contains the data to plot
    :param within_column: (string) name of the column containing the within subject labels. This should contain in
    most cases the subject ID but in general whatever unit the measurement was repeated in
    :param between_column: (string) name of the column containing the label of the experimental condition for which
    we want to display the variance between.
    :param dependent_variable: (string) name of the column containing the dependent variable to plot (reaction time...)
    :param ax: (matplotlib ax object) ax on which to plot the image.
    :param positions: (None or string or array-like) x axis position for the boxplot. If string,the value within the
    data frame of that particular column are taken as position. The value in that column must be numerical. Handy if you
    have a column encoding a continuous value that you want to plot along.
    :param cousineau_correction: (bool) whether to apply cousineau Morey correction.
    :param title: (string) title of the plot
    :param xlabel: (string) xlabel
    :param ylabel: (string) ylabel
    :param xlim: (list) limits for the x axis
    :return:
    """
    if cousineau_correction:
        data_df = cousineau_morey_correction(data_df, within_column, between_column, dependent_variable)

    # Average the data within subject and condition for the boxplot:
    avg_data = data_df.groupby([within_column, between_column])[dependent_variable].mean().reset_index()
    # Convert to 2D arrays for the line plot:
    try:
        avg_data_2d = np.array([avg_data[avg_data[between_column] == cond][dependent_variable].to_numpy()
                                for cond in avg_data[between_column].unique()]).T
    except ValueError:
        avg_data_2d = np.zeros((len(avg_data[within_column].unique()), len(avg_data[between_column].unique())))
        for sub_i, sub in enumerate(avg_data[within_column].unique()):
            for cond_i, cond in enumerate(avg_data[between_column].unique()):
                try:
                    avg_data_2d[sub_i, cond_i] = avg_data.loc[(avg_data[within_column] == sub) &
                                                              (avg_data[between_column] == cond),
                    dependent_variable].values
                except ValueError:
                    print("WARNING: missing value for sub-{} in condition {}".format(sub, cond))
                    avg_data_2d[sub_i, cond_i] = np.nan

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=[14, 14])
    if isinstance(positions, str):
        positions = avg_data[between_column].unique()
        if isinstance(positions[0], str):
            positions = [float(pos) for pos in positions]
    elif positions is None:
        positions = range(0, len(avg_data[between_column].unique()))
    # Check if there are NaNs in the data:
    if np.isnan(np.min(avg_data_2d)):
        nan_inds = np.where(np.isnan(avg_data_2d))
        for row_ind in nan_inds[0]:
            for col_ind in nan_inds[1]:
                avg_data_2d[row_ind, col_ind] = np.mean(avg_data_2d[col_ind])
    bplot = ax.boxplot(avg_data_2d, patch_artist=True, notch=False,
                       positions=positions, widths=width, medianprops=dict(color="black", linewidth=1.5))
    lineplot = ax.plot(positions, avg_data_2d.T, ':', linewidth=0.5,
                       color=[0.5, 0.5, 0.5], alpha=0.5)
    ax.tick_params(axis='x', labelrotation=45, labelsize=xlabel_fontsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)

    if face_colors is not None:
        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(face_colors[i])
    if edge_colors is not None:
        for i, patch in enumerate(bplot['boxes']):
            patch.set_edgecolor(edge_colors[i])
    return ax, bplot, lineplot


def soa_boxplot(data_df, dependent_variable, fig_size=None, lock_column="SOA_lock", subject_column="sub_id",
                between_column="onset_SOA", ax=None, fig=None, colors_onset_locked=None, colors_offset_locked=None):
    """
    This function plots the PRP study data in a standardized format, so that it can be used across experiments and data
    types. It is not super well documented, but it is not meant to be reuused as highly specific to this design.
    :param between_column:
    :param subject_column:
    :param lock_column:
    :param dependent_variable:
    :param data_df:
    :param fig_size:
    :return:
    """
    if fig_size is None:
        fig_size = [8.3 / 3, 11.7 / 2]
    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, figsize=fig_size)
    if colors_onset_locked is None:
        colors_onset_locked = [val for val in ev.colors["soa_onset_locked"].values()]
    if colors_offset_locked is None:
        colors_offset_locked = [val for val in ev.colors["soa_offset_locked"].values()]
    d = 1.5
    # Onset locked data:
    _, _, _ = plot_within_subject_boxplot(data_df[data_df[lock_column] == 'onset'],
                                          subject_column, between_column, dependent_variable,
                                          positions=between_column, ax=ax[0], cousineau_correction=True,
                                          title="",
                                          xlabel="", ylabel="",
                                          xlim=[-0.1, 0.6], width=0.1,
                                          face_colors=colors_onset_locked)
    # Loop through each duration to plot the offset locked SOA separately:
    for i, dur in enumerate(sorted(list(data_df["duration"].unique()))):
        _, _, _ = plot_within_subject_boxplot(data_df[(data_df[lock_column] == 'offset')
                                                      & (data_df["duration"] == dur)], subject_column,
                                              between_column,
                                              dependent_variable,
                                              positions=between_column, ax=ax[i + 1], cousineau_correction=True,
                                              title="",
                                              xlabel="", ylabel="",
                                              xlim=[dur - 0.1, dur + 0.6], width=0.1,
                                              face_colors=colors_offset_locked)
        ax[i + 1].yaxis.set_visible(False)
    # Remove the spines:
    for i in [0, 1, 2]:
        ax[i].spines['right'].set_visible(False)
        ax[i + 1].spines['left'].set_visible(False)
        # Add cut axis marks:
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax[i].plot([1, 1], [0, 0], transform=ax[i].transAxes, **kwargs)
        ax[i + 1].plot([0, 0], [0, 0], transform=ax[i + 1].transAxes, **kwargs)
        ax[i].plot([1, 1], [1, 1], transform=ax[i].transAxes, **kwargs)
        ax[i + 1].plot([0, 0], [1, 1], transform=ax[i + 1].transAxes, **kwargs)
    plt.subplots_adjust(wspace=0.05)

    return fig, ax


def plot_ts_ci(data, times, color, plot_ci=True, ax=None, label="", clusters=None, clusters_pval=None,
               clusters_alpha=0.3, sig_thresh=0.01, plot_nonsig_clusters=False, cluster_color="r"):
    """

    :param data:
    :param times:
    :param color:
    :param plot_ci:
    :param ax:
    :param label:
    :param clusters:
    :param clusters_pval:
    :param clusters_alpha:
    :param sig_thresh:
    :param plot_nonsig_clusters:
    :param cluster_color:
    :return:
    """
    assert len(data.shape) == 2, "The data must be time series and of shape n_obs x samples!"
    assert data.shape[1] == times.shape[0], "The time vector and the data do not have the same length!"
    if ax is None:
        fig, ax = plt.subplots()
    # Average across observation:
    data_avg = np.mean(data, axis=0)
    # Compute bootstrapped confidence interval for each condition:
    data_ci = bootstrap_confidence_interval(data)
    # Plot the evoked:
    ax.plot(times, data_avg, label=label,
            color=color)
    # Plot the CI
    if plot_ci:
        ax.fill_between(times, data_ci[0, :], data_ci[1, :],
                        color=color,
                        alpha=.1)
    if clusters is not None:
        for i_c, c in enumerate(clusters):
            c = c[0]
            if clusters_pval[i_c] <= sig_thresh:
                h = ax.axvspan(times[c.start], times[c.stop - 1], color=cluster_color, alpha=clusters_alpha)
            elif clusters_pval[i_c] <= 0.05:
                if plot_nonsig_clusters:
                    ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=clusters_alpha)
    ax.set_xlim([times[0], times[-1]])
    return ax


def plot_pupil_latency(evoked_dict, times, latencies_df, colors, boxplot_ylim=None):
    """

    :param evoked_dict:
    :param latencies_df:
    :param times:
    :param colors:
    :param boxplot_ylim:
    :return:
    """
    fig = plt.figure(figsize=[10, 11.7 / 4], constrained_layout=True)
    spec = fig.add_gridspec(ncols=4, nrows=1)
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[0, 3])

    for soa in evoked_dict.keys():
        # Plot the evoked pupil response:
        plot_ts_ci(np.array(evoked_dict[soa]), times, colors[soa], plot_ci=False, ax=ax1, label=soa)
        # Extract the mean latencies:
        lat = np.mean(latencies_df[latencies_df["SOA"] == soa]["latency"].to_numpy())
        # Plot the latency:
        ax1.vlines(x=latencies_df[latencies_df["SOA"] == soa]["SOA_locked"].to_numpy()[0],
                   ymin=-0.02, ymax=-0.01, linestyle="-",
                   color=ev.colors["soa_onset_locked"][soa], linewidth=2, zorder=10)
        ax1.vlines(x=lat, ymin=-0.02,
                   ymax=np.mean(np.array(evoked_dict[soa]), axis=0)[np.argmin(np.abs(times - lat))],
                   linestyle="--",
                   color=colors[soa], linewidth=2, zorder=10)
    ax1.set_ylim([-0.02, 0.09])
    ax1.legend()
    ax1.set_xlabel("Time (sec.)")
    ax1.set_ylabel("Pupil size (norm.)")
    # Plot the latency as a function of SOA:
    plot_within_subject_boxplot(latencies_df, "sub_id", "SOA", "latency_aud",
                                positions="SOA", ax=ax2, cousineau_correction=False, title="", xlabel="SOA",
                                ylabel=r'$\tau_{\mathrm{audio}}$', xlim=[-0.1, 0.6], width=0.1,
                                face_colors=[colors[soa] for soa in list(evoked_dict.keys())])
    ax2.set_ylim(boxplot_ylim)
    plt.suptitle("Pupil peak latency")
    return fig

