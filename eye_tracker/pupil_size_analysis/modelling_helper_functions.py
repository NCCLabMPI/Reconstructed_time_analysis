import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
import pandas as pd
from statsmodels.regression.linear_model import OLS
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']


def data_plotter(data, times, trial_matrix, ax=None, show_legend=True, plot_data=True, plot_predicted=True,
                 plot_timings=True, plot_purf=True):
    """"
    This function plots the data with the events
    :param data: (numpy array) data to plot
    :param times: (numpy array) time vector
    :param trial_matrix: (pandas dataframe) trial matrix
    :param ax: (matplotlib axis) axis to plot on
    :return: (matplotlib axis) axis with the plot
    """

    # Create the axis if not provided:
    if ax is None:
        fig, ax = plt.subplots(nrows=data.shape[0], ncols=1, sharex=True, sharey=True)
    # Create the canonical PURF:
    can_purf = purf(times, tmax=0.93, n=10.1)
    # Loop over the trials:
    for trial in range(data.shape[0]):
        # Plot the data:
        if plot_data:
            ax[trial].plot(times, data[trial, :], color="grey", alpha=0.5, label="Data")
        if plot_timings:
            # Add vertical lines to mark the onset of the events according to the trial matrix:
            pos = np.linspace(0, 1, len(trial_matrix.columns), endpoint=False)
            inter = (pos[1] - pos[0]) / 2
            for i, evt in enumerate(trial_matrix.columns):
                ax[trial].axvline(trial_matrix[evt][trial], ymin=pos[i], ymax=pos[i] + inter, color=cwheel[i], linewidth=2)
        if plot_purf:
            for i, evt in enumerate(trial_matrix.columns):
                # Add the purf:
                evt_purf = np.convolve(create_sticks(times, trial_matrix[evt][trial]), can_purf,
                                       mode="full")[0:len(times)]
                ax[trial].plot(times, evt_purf, color=cwheel[i], linestyle=":", label=evt)

        if plot_predicted:
            # Plot the predicted response:
            ax[trial].plot(times, experiment_model(data[trial, :], times,
                                                   *trial_matrix.iloc[trial].to_list()).predict(),
                           color=cwheel[1], linestyle="--", label="Predicted response")
        # Remove the left and top spines:
        ax[trial].spines['right'].set_visible(False)
        ax[trial].spines['top'].set_visible(False)
        ax[trial].set_xlabel("Time (s)")
        ax[trial].set_ylabel("Pupil size")
        ax[trial].set_xlim([times[0], times[-1]])

    if show_legend:
        plt.legend()

    return ax


def estimate_latencies(data, times, trial_matrix, bounds_range, n_iterations=100, n_initial_values=200, n_best_values=40):
    """

    :param data:
    :param times:
    :param trial_matrix:
    :return:
    """
    # Create a list to store the results:
    estimated_latencies = []
    # Loop through each trial:
    for trial in range(trial_matrix.shape[0]):
        trial_estimated_latencies = pd.DataFrame()
        # Repeat the procedure n times:
        print("Estimation for trial {}".format(trial))
        for i in tqdm(range(n_iterations)):
            # Extract the values from all columns:
            events_timings = trial_matrix.iloc[trial, :].to_list()
            # Generate the bounds:
            try:
                bounds = [[evt - bound, evt + bound] for evt, bound in zip(events_timings, bounds_range)]
            except TypeError:
                print("Weird")
            # Generate the initial values:
            initial_values = np.array([np.random.uniform(bound[0], bound[1], n_initial_values) for bound in bounds])
            objective_values = []
            # Generate the objective function:
            for i in range(initial_values.shape[1]):
                objective_values.append(fitting_function(initial_values[:, i],
                                                         data[trial, :], times))
            # Take the best initial values:
            best_initial_values = initial_values[:, np.argpartition(objective_values, n_best_values)[0:40]]

            # Perform optimization with these starting values:
            optmization_results = []
            for i in range(best_initial_values.shape[1]):
                # Perform optimization with these starting values:
                optmization_results.append(minimize(fitting_function, best_initial_values[:, i],
                                                    args=(data[trial, :], times),
                                                    bounds=bounds))
            # Get the best results:
            best_results = optmization_results[np.argmin([result.fun for result in optmization_results])]
            # Append to the estimated table:
            trial_estimated_latencies = pd.concat([trial_estimated_latencies,
                                                   pd.DataFrame({col: val for col, val in
                                                                 zip(trial_matrix.columns, best_results.x)},
                                                                index=[0])],
                                                  axis=0)
        trial_estimated_latencies["trial"] = trial
        estimated_latencies.append(trial_estimated_latencies)

    # Concatenate everything:
    estimated_latencies = pd.concat(estimated_latencies).reset_index(drop=True)
    # Return the results:
    return estimated_latencies


def fitting_function(free_params, *args):
    # Unpack the free parameters:
    latency_vis, latency_audio, latency_resp, latency_offset = free_params
    # Unpack the remaining parameters:
    [data, times] = args

    # Create the model:
    results = experiment_model(data, times, latency_vis, latency_audio, latency_resp, latency_offset)

    return 1 - results.rsquared


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


def purf(times, tmax, n=10.1):
    output = (times ** n) * np.exp(-n * times / tmax)
    output[times < 0] = 0
    return output / np.max(output)


def simulator(times, trial_matrix, tmax=0.93, n=10.1):
    # Generate the canonical purf:
    can_purf = purf(times, tmax=tmax, n=n)
    data = np.zeros((trial_matrix.shape[0], len(times)))
    # Loop through each trial:
    for trial in range(trial_matrix.shape[0]):
        # Loop through each event:
        for evt in trial_matrix.columns:
            # Get the event time:
            evt_time = trial_matrix[evt][trial]
            # Create the stick function:
            stick = create_sticks(times, evt_time)
            # Convolve the stick with the canonical purf:
            data[trial, :] += (np.convolve(stick, can_purf, mode="full")[0:len(times)] +
                               np.random.normal(0, 0.1,
                                                len(times)))
    return data


def experiment_model(data, times, *args):
    # Create the canonical PURF:
    can_purf = purf(times, tmax=0.93, n=10.1)
    # Create the regressors for each argument:
    exog = np.zeros((len(args), len(times)))
    for arg_ind, arg in enumerate(args):
        # Create the stick:
        stk = create_sticks(times, arg)
        # Convolve the stick with the canonical PURF:
        exog[arg_ind, :] = np.convolve(stk, can_purf, mode="full")[0:len(times)]
    # Model the data:
    results = OLS(data, exog.T).fit()
    return results
