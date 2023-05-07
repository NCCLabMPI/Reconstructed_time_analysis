from mne.baseline import rescale


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    This function performs baseline correction on the data. The default is to compute the mean over the entire baseline
    and dividing each data points in the entire epochs by it. Another option is to substract baseline from each time
    point
    :param epochs: (mne epochs object) epochs on which to perform the baseline correction
    :param correction_method: (string) options to do the baseline correction. Options are:
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
          source: https://github.com/mne-tools/mne-python/blob/main/mne/baseline.py
    :param baseline: (tuple) which bit to take as the baseline
    :param picks: (None or list of int or list of strings) indices or names of the channels on which to perform the
    correction. If none, all channels are used
    :param n_jobs: (int) number of jobs to use to preprocessing the function. Can be ran in parallel
    :return: none, the data are modified in place
    """
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None

