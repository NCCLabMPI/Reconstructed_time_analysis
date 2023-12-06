import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np
from general_helper_function import get_event_ts

font = {'size': 18}
matplotlib.rc('font', **font)
# Get matplotlib colors:
prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']


def plot_events_latencies(epochs, lock, durations, soas, channels=None, audio_lock=True):
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
                                ax=None, cousineau_correction=True, title="", xlabel="", ylabel=""):
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
    :return:
    """
    if cousineau_correction:
        data_df = cousineau_correction(data_df, within_column, between_column, dependent_variable)

    # Average the data within subject and condition for the boxplot:
    avg_data = data_df.groupby([within_column, between_column])[dependent_variable].mean().reset_index()
    # Convert to 2D arrays for the line plot:
    avg_data_2d = np.array([avg_data[avg_data[cond] == cond][dependent_variable].to_numpy()
                            for cond in avg_data[between_column].unique()]).T

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=[14, 14])
    if isinstance(positions, str):
        positions = avg_data[between_column].unique()
    elif positions is None:
        positions = range(0, len(avg_data[between_column].unique()))
    bplot = ax.boxplot(avg_data_2d, patch_artist=True, notch=True,
                       positions=positions, widths=0.1)
    lineplot = ax.plot(positions, avg_data_2d.T, linewidth=0.6,
                       color=[0.5, 0.5, 0.5], alpha=0.5)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax, bplot, lineplot
