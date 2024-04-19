import numpy as np
import pandas as pd
from math import floor
from scipy.optimize import minimize
import statsmodels.api as sm
from tqdm import tqdm
from numpy.random import normal, uniform
import matplotlib.pyplot as plt


prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']


def purf_fun(times, tmax, n=10.1):
    output = (times ** n) * np.exp(-n * times / tmax)
    output[times < 0] = 0
    return output / np.max(output)


def create_sticks(t, event_time):
    """
    This function creates a stick predictor for each event time
    :param t: (numpy array) time vector
    :param event_time: (list) list of event times
    :return: (numpy array) stick predictor
    """
    # Create the stick predictor:
    stick = np.zeros(len(t))
    stick[np.argmin(np.abs(t - event_time))] = 1
    return stick


def create_stick_predictors(design_matrix, times):
    """

    :param design_matrix:
    :param times:
    :return:
    """
    # Extract dimensions:
    ntrials = design_matrix.shape[0]
    npred = design_matrix.shape[1]
    ntimes = times.shape[0]
    dt = times[1] - times[0]

    # Preallocate the stick matrix:
    sticks_matrix = np.zeros([len(times) * ntrials, npred])
    # Loop through each column to create the stick predictors:
    for col_ind, col in enumerate(design_matrix.columns):
        # Extract the time stamps in this column:
        time_stamps = design_matrix[col].to_numpy()
        # Prepare matrix of zeros to store the sticks:
        col_stick = np.zeros([ntimes, ntrials])
        # Loop through each trial:
        for evt_i, evt_time in enumerate(time_stamps):
            # In case of Nan, keeping the whole time series to 0s:
            if np.isnan(evt_time):
                continue
            # Find the nearest index for the current time stamp
            nearest_time_ind = int((evt_time - times[0]) / (times[1] - times[0]))
            # Set the corresponding value to 1:
            col_stick[nearest_time_ind, evt_i] = 1
        # We now have a matrix of dimension times * trial. We will now concatenate it across trials:
        sticks_concat = np.reshape(col_stick.T, (np.product(col_stick.shape)))
        # Add to the preallocated matrix:
        sticks_matrix[:, col_ind] = sticks_concat

    # Finally, create the concatenated time vector:
    times_concat = np.linspace(times[0], times[0] + (sticks_matrix.shape[0] - 1) * dt, sticks_matrix.shape[0])

    return sticks_matrix, times_concat


def create_design_matrix(timing_mat, cond_mat, lat_mat, purf, times, verbose=True, add_jitter=0):
    """

    :param timing_mat:
    :param cond_mat:
    :param lat_mat:
    :param purf:
    :param times:
    :param verbose:
    :return:
    """
    if verbose:
        print("="*40)
        print("Creating design matrix")
    # Check inputs:
    ncond = cond_mat.shape[1]
    nevents = timing_mat.shape[1]
    if ncond * nevents != lat_mat.shape[0]:
        raise ValueError("The latency matrix does not contain one value per event x condition!")
    # Adjust the condition matrix to replace 0s by nan. Otherwise, when creating the stick predictors, trials in which
    # a particular event does not occur will be interpreted as a latency 0, which would be a valid input:
    cond_mat[np.where(cond_mat == 0)] = np.nan
    # Multiply the condition matrix with the timing matrix, to obtain one column per condition x events. That way
    # we have the timing of each event separated per experimental condition:
    timing_x_cond_mat = []
    for i in range(cond_mat.shape[1]):
        timing_x_cond_mat.append(np.transpose(cond_mat[:, i] * timing_mat.T))
    timing_x_cond_mat = np.concatenate(timing_x_cond_mat, axis=1)

    # Multiply the latency matrix by the condition matrix, to obtain one latency value per trial, separately for each
    # condition
    lat_x_cond_mat = []
    for i in range(lat_mat.shape[0]):
        lat_x_cond_mat.append(lat_mat[i] * cond_mat[:, floor(i / (lat_mat.shape[0] / cond_mat.shape[1]))])
    lat_x_cond_mat = np.array(lat_x_cond_mat).T

    # Adjust the timing_x_cond_mat by adding the latencies:
    adj_timing_x_cond_mat = timing_x_cond_mat + lat_x_cond_mat
    if add_jitter > 0:
        jitters = normal(0, add_jitter, adj_timing_x_cond_mat.shape)
        # Ensure that the jitters are strictly positive:
        adj_timing_x_cond_mat = adj_timing_x_cond_mat + (jitters + np.max(jitters))
    # Create the stick predictors based of the adj_timing_x_cond_mat:
    sticks_predictors, times_concat = create_stick_predictors(pd.DataFrame(adj_timing_x_cond_mat), times)

    # Apply convolution between the adj_timing_x_cond_mat and the kernel:
    design_matrix = np.array([np.convolve(sticks_predictors[:, i], purf, mode='full')[0:sticks_predictors.shape[0]]
                              for i in range(sticks_predictors.shape[1])]).T
    # Add intercept:
    design_matrix = np.append(design_matrix, np.ones([1, design_matrix.shape[0]]).T, 1)

    return design_matrix, times_concat


def latency_glm(data, kernel, timing_mat, cond_mat, times, range_bounds=None, regressor_names=None,
                n_initial_values=10):
    """

    :param kernel:
    :param data:
    :param timing_mat:
    :param cond_mat:
    :param times:
    :param range_bounds:
    :param regressor_names:
    :param n_initial_values:
    :return:
    """
    print("="*40)
    print("Estimating latency parameters")
    # Handle inputs:
    if range_bounds is None:
        range_bounds = [[0, 0.5, 10]] * (timing_mat.shape[1] * cond_mat.shape[1])
    if regressor_names is None:
        regressor_names = ["reg-{}".format(i) for i in range(timing_mat.shape[1] * cond_mat.shape[1] + 1)]

    # ============================================================
    # 1. Finding initial values:
    # Specify the search range for each parameter:
    search_ranges = [uniform(val[0], val[1], val[2]) for val in range_bounds]
    # Create the search mat as a meshgrid to get the combinations for each parameters:
    search_mat = np.meshgrid(*search_ranges)
    # Reshape so that each column contains a set of latency value for each parameter:
    search_mat = np.vstack([x.ravel() for x in search_mat]).T
    # Loop through each:
    rsquared_exp = []
    expl_results = []
    for lats in tqdm(search_mat):
        # Fit the GLM with this set of parameters:
        res = fit_glm(data, timing_mat, cond_mat, lats, kernel, times)
        # Store the results to a dataframe:
        keys = (["beta:{}".format(reg) for reg in regressor_names] +
                ["lat:{}".format(reg) for reg in regressor_names[0:-1]] +
                ["rsquared"])
        vals = list(res.params) + list(lats) + [res.rsquared]
        expl_results.append(pd.DataFrame(
            dict(zip(keys, vals)), index=[0]
        ))
        rsquared_exp.append(-res.rsquared)
    expl_results = pd.concat(expl_results).sort_values("rsquared").reset_index(drop=True)
    # Extract the best latency:
    initial_values = search_mat[np.argsort(np.array(rsquared_exp))[0:n_initial_values]]

    # ============================================================
    # 2. Optimization from best guesses:
    # Loop through each for the optimization:
    optimized_lat = []
    optimized_rsquared = []
    for init_lats in initial_values:
        opt_results = minimize(cost_function, init_lats,
                               args=(timing_mat, cond_mat, kernel, times, data),
                               bounds=[(bnd[0], bnd[1]) for bnd in range_bounds])
        optimized_lat.append(opt_results.x)
        optimized_rsquared.append(opt_results.fun)
    optimized_lat = np.array(optimized_lat)
    optimized_rsquared = np.array(optimized_rsquared)

    # ============================================================
    # 3. Fitting model with best fit:
    best_lats = optimized_lat[np.argmin(optimized_rsquared)]
    results = fit_glm(data, timing_mat, cond_mat, best_lats, kernel, times)
    keys = (["beta:{}".format(reg) for reg in regressor_names] +
            ["lat:{}".format(reg) for reg in regressor_names[0:-1]] +
            ["rsquared"])
    vals = list(results.params) + list(best_lats) + [optimized_rsquared[np.argmin(optimized_rsquared)]]
    best_fit_results = pd.DataFrame(
        dict(zip(keys, vals)), index=[0]
    )

    # Return the data frame sorted per R squared:
    return best_lats, best_fit_results, expl_results


def cost_function(lats, timing_mat, cond_mat, kernel, times, data):
    """

    :param lats:
    :param timing_mat:
    :param cond_mat:
    :param kernel:
    :param times:
    :param data:
    :return:
    """
    # Fit the GLM model to the data:
    results = fit_glm(data, timing_mat, cond_mat, lats, kernel, times)
    return -results.rsquared  # Minimize negative of R-squared to maximize R-squared


def fit_glm(data, timing_mat, cond_mat, lats, kernel, times):

    # Create the design matrix with this set of parameters:
    design_matrix_opt, times_concat = create_design_matrix(timing_mat, cond_mat,
                                                           lats, kernel, times, verbose=False, add_jitter=0)
    # Fit GLM:
    model = sm.OLS(data, design_matrix_opt)
    results = model.fit()

    return results


def plot_glm(data, times, fitted_data, design_matrix, regressor_names, axs=None, decorate_axes=True, clear_axes=False):
    """

    :param data:
    :param times:
    :param fitted_data:
    :param design_matrix:
    :param regressor_names:
    :param axs:
    :param decorate_axes:
    :return:
    """

    if axs is None:
        fig, axs = plt.subplots(2, 1, layout='constrained')
    if clear_axes:
        for ax in axs:
            while ax.lines:
                ax.lines[0].remove()
    # Plot the observed data:
    axs[0].plot(times, data, label="Data", color="blue")
    # Plot the fitted data:
    axs[0].plot(times, fitted_data, label="Fitted", color="red")

    # Plot the computed design matrix:
    [axs[1].plot(times, design_matrix[:, i], label=regressor_names[i], color=cwheel[i])
     for i in range(design_matrix.shape[1])]

    if decorate_axes:
        # Set labels:
        axs[0].set_ylabel("Pupil size")
        axs[0].set_title("Fitted data")
        axs[0].spines[['right', 'top']].set_visible(False)
        axs[1].set_title("Computed design matrix")
        axs[1].spines[['right', 'top']].set_visible(False)
        plt.legend(fontsize="xx-small", loc="upper right")
        plt.tight_layout()

    return axs


def plot_parameters_comparison(true_betas, obs_betas, true_lat, obs_lat, regressor_names, ax=None):
    """

    :param true_betas:
    :param obs_betas:
    :param true_lat:
    :param obs_lat:
    :param regressor_names:
    :param ax:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, layout='constrained')
    x = np.arange(len(true_betas))  # the label locations
    width = 0.4  # the width of the bars
    # Plot the betas ground truth:
    ax[0].bar(x - 0.2, true_betas, width, label="True", color=[0, 0, 0])
    # Plot the computed betas:
    ax[0].bar(x + 0.2, obs_betas, width, label="Retrieved", color=[0.5, 0.5, 0.5])
    # Add some text:
    ax[0].set_ylabel('Betas')
    ax[0].set_title('Computed vs ground truth betas')
    ax[0].set_xticks(x, regressor_names)
    ax[0].legend()
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=45, ha='right')
    # Same for the latency parameters:
    x = np.arange(len(true_lat))  # the label locations
    ax[1].bar(x - 0.2, true_lat, width, label="True", color=[0, 0, 0])
    # Plot the computed latencies:
    ax[1].bar(x + 0.2, obs_lat, width, label="Retrieved", color=[0.5, 0.5, 0.5])
    # Add some text:
    ax[1].set_ylabel('Latency (sec.)')
    ax[1].set_title('Computed vs ground truth latencies')
    ax[1].set_xticks(x, regressor_names[0:-1])
    ax[1].legend()
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=45, ha='right')

    return ax
