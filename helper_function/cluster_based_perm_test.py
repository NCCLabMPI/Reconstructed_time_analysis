def cluster_test(x_obs, null_dist, z_threshold=None, adjacency=None, tail=1, max_step=None, exclude=None,
                 t_power=1, step_down_p=0.05, do_zscore=True):
    """
    This function performs a cluster based permutation test on a single observation array with respect to a null
    distribution. This is useful in case where for example decoding was performed on a single subject and a null
    distribution was obtained by shuffling the labels and performing the analysis a 1000 times. You are left with one
    array of observed value and several arrays constituting your null distribution. In a classical cluster based
    permutation test, a statistical test will be performed and cluster-summed, and then compared to cluster-sum values
    obtained by shuffling the observation across groups. In this case, because there is only one observation, that
    doesn't work. Instead, the observed data and the null distribution get z scored. Then, cluster sum are computed
    both on the x and h0 to assess which clusters are significant.
    NOTE: this was created by selecting specific bits from:
    https://github.com/mne-tools/mne-python/blob/eb14b9c55c65573a27624533e9224dcf474f6ad5/mne/stats/cluster_level.py#L684
    :param x_obs: (1 or 2D array) contains the observed data for which to compute the cluster based permutation test
    :param null_dist: (x.ndim + 1 array) contains the null distribution associated with the observed data. The dimensions
    must be as follows: [n, p, (q)] where n are the number of observation (i.e. number of permutation that were used
    to generate the null distribution), p and (q) correspond to the dimensions of the observed data
    (time and frequency, or only time, or time x time...)
    :param z_threshold: (float) z score threshold for something to be considered eligible for a cluster
    :param adjacency: (scipy.sparse.spmatrix | None | False) see here for details:
    https://mne.tools/stable/generated/mne.stats.spatio_temporal_cluster_test.html
    :param tail: (int) 1 for upper tail, -1 lower tail, 0 two tailed
    :param max_step: (int) see here for details:
    https://mne.tools/stable/generated/mne.stats.spatio_temporal_cluster_test.html
    :param exclude: (bool array or None) array of same dim as x for excluding specific parts of the matrix from analysis
    :param t_power: (float) power by which to raise the z score by. When set to 0, will give a count of locations in
    each cluster, t_power=1 will weight each location by its statistical score.
    :param step_down_p: (float) To perform a step-down-in-jumps test, pass a p-value for clusters to exclude from each
    successive iteration.
    :param do_zscore: (boolean) if the data are zscores already, don't redo the z transform
    :return:
    x_zscored: (x.shape np.array) observed values z scored
    h0_zscore: (h0.shape np.array) null distribution values z scored
    clusters: (list) List type defined by out_type above.
    cluster_pv: (array) P-value for each cluster.
    p_values: (x.shape np.array) p value for each observed value
    H0: (array) Max cluster level stats observed under permutation.
    """
    print("=" * 40)
    print("Welcome to cluster_test")
    # Checking the dimensions of the two input matrices:
    if x_obs.shape != null_dist.shape[1:]:
        raise Exception("The dimension of the observed matrix and null distribution are inconsistent!")

    # Get the original shape:
    sample_shape = x_obs.shape
    # Get the number of tests:
    n_tests = np.prod(x_obs.shape)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')
    # Step 1: Calculate z score for original data
    # -------------------------------------------------------------
    if do_zscore:
        print("Z scoring the data:")
        x_zscored = zscore_mat(x_obs, null_dist, axis=0)
        h0_zscore = [zscore_mat(null_dist[i], np.append(x_obs[None], null_dist, axis=0)) for i in range(null_dist.shape[0])]
    else:
        x_zscored = x_obs
        h0_zscore = [null_dist[i] for i in range(null_dist.shape[0])]

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # Step 2: Cluster the observed data:
    # -------------------------------------------------------------
    print("Finding the cluster in the observed data:")
    out = _find_clusters(x_zscored, z_threshold, tail, adjacency,
                         max_step=max_step, include=include,
                         partitions=None, t_power=t_power,
                         show_info=True)
    clusters, cluster_stats = out

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        clusters = _cluster_indices_to_mask(clusters, 20)

    # Compute the clusters for the null distribution:
    if len(clusters) == 0:
        print('No clusters found, returning empty H0, clusters, and cluster_pv')
        return x_zscored, h0_zscore, np.array([]), np.array([]), np.array([]), np.array([])

    # Step 3: repeat permutations for step-down-in-jumps procedure
    # -------------------------------------------------------------
    n_removed = 1  # number of new clusters added
    total_removed = 0
    step_down_include = None  # start out including all points
    n_step_downs = 0
    print("Finding the cluster in the null distribution:")
    while n_removed > 0:
        # actually do the clustering for each partition
        if include is not None:
            if step_down_include is not None:
                this_include = np.logical_and(include, step_down_include)
            else:
                this_include = include
        else:
            this_include = step_down_include
        # Find the clusters in the null distribution:
        _, surr_clust_sum = zip(*[_find_clusters(mat, z_threshold, tail, adjacency,
                                                 max_step=max_step, include=this_include,
                                                 partitions=None, t_power=t_power,
                                                 show_info=True) for mat in h0_zscore])
        # Compute the max of each surrogate clusters:
        h0 = [np.max(arr) if len(arr) > 0 else 0 for arr in surr_clust_sum]
        # Get the original value:
        if tail == -1:  # up tail
            orig = cluster_stats.min()
        elif tail == 1:
            orig = cluster_stats.max()
        else:
            orig = abs(cluster_stats).max()
        # Add the value from the original distribution to the null distribution:
        h0.insert(0, orig)
        h0 = np.array(h0)
        # Extract the p value of the max cluster by locating the observed cluster sum on the surrogate cluster sums:
        cluster_pv = _pval_from_histogram(cluster_stats, h0, tail)

        # figure out how many new ones will be removed for step-down
        to_remove = np.where(cluster_pv < step_down_p)[0]
        n_removed = to_remove.size - total_removed
        total_removed = to_remove.size
        step_down_include = np.ones(n_tests, dtype=bool)
        for ti in to_remove:
            step_down_include[clusters[ti]] = False
        if adjacency is None and adjacency is not False:
            step_down_include.shape = sample_shape
        n_step_downs += 1

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(x_obs).T
    for cluster, pval in zip(clusters, cluster_pv):
        if isinstance(cluster, np.ndarray):
            p_values_[cluster.T] = pval
        elif isinstance(cluster, tuple):
            p_values_[cluster] = pval

    return x_zscored, h0_zscore, clusters, cluster_pv, p_values_.T, h0


def zscore_mat(x, h0, axis=0):
    """
    This function computes a zscore between a value x and a
    :param x: (float) a single number for which to compute the zscore with respect ot the y distribution to the
    :param h0: (1d array) distribution of data with which to compute the std and mean:
    :param axis: (int) which axis along which to compute the zscore for the null distribution
    :return: zscore
    """
    assert isinstance(x, np.ndarray) and isinstance(h0, np.ndarray), "x and y must be numpy arrays!"
    assert len(h0.shape) == len(x.shape) + 1, "y must have 1 dimension more than x to compute mean and std over!"
    try:
        zscore = (x - np.mean(h0, axis=axis)) / np.std(h0, axis=axis)
    except ZeroDivisionError:
        Exception("y standard deviation is equal to 0, can't compute zscore!")

    return zscore




# Example of how to plot the results:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x_pix = 1920
y_pix = 1080

x_obs = np.random.normal(size=[200, 200])
x_null = np.random.normal(size=[1000, 200, 200])

x_obs_z, x_null_z, clusters, cluster_pv, p_values, H0  =cluster_test(x_obs, x_null, z_threshold=1.96, adjacency=None, tail=1, max_step=None, exclude=None,
                 t_power=1, step_down_p=0.05, do_zscore=True)
sig_mask = x_obs_z.copy()
sig_mask[p_values > 0.05] = np.nan
norm = matplotlib.colors.TwoSlopeNorm(vmin=min(x_obs_z), vcenter=0, vmax=max(x_obs_z))

fig, ax = plt.subplots(figsize=[20, 15])
# Plot matrix with transparency:
im = ax.imshow(x_obs_z, cmap='Reds', norm=norm,
                extent=[0, x_pix, 0, y_pix],
                origin="lower", alpha=0.4, aspect='equal')
# Plot the significance mask on top:
if not np.isnan(sig_mask).all():
    # Plot only the significant bits:
    im = ax.imshow(sig_mask, cmap='Reds', origin='lower', norm=norm,
                    extent=[0, x_pix, 0, y_pix],
                    aspect='equal')
# Add the axis labels and so on:
ax.set_xlabel('X (pix)')
ax.set_ylabel('Y (pix)')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.tight_layout()
cb = plt.colorbar(im)
if DO_ZSCORE:
    cb.ax.set_ylabel('Z score')
else:
    cb.ax.set_ylabel('Correlation difference within vs between')
cb.ax.yaxis.set_label_position('left')
# Finally, adding the significance contour:
if not np.isnan(sig_mask).all():
    ax.contour(sig_mask > 0, sig_mask > 0, colors="k", origin="lower",
                extent=[0, x_pix, 0, y_pix])


