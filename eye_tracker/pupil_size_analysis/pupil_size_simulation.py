import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import pandas as pd
from eye_tracker.pupil_size_analysis.pupil_size_helper_functions import (purf_fun, create_design_matrix, latency_glm,
                                                                         plot_glm, plot_parameters_comparison)

prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']

# Set fixed parameters:
SFREQ = 100
tmin = -0.2
tmax = 4.0
TIMES = np.arange(tmin, tmax, 1 / SFREQ)
PURF_TMAX = 0.93
PURF_N = 10.1
# Set the number of events per trials:
NEVT_PER_TRIAL = 4
EVT_NAMES = ["visOnset", "visOffset", "aud", "rt"]
N_CONDITIONS = 2
CONDITIONS_NAMES = ["soa1", "soa2"]


def run_simulation(jitter, ntrials_per_cond, noise, beta_noise, show_plots=True):
    # Create pupil response function kernel
    purf = purf_fun(TIMES[TIMES > 0], PURF_TMAX, n=PURF_N)
    # =========================================
    # Generate beta value for each parameter:
    true_betas = pd.DataFrame({
        "visOnset-soa1": normal(0.8, beta_noise, 1)[0],
        "visOffset-soa1": normal(0.5, beta_noise, 1)[0],
        "aud-soa1": normal(0.2, beta_noise, 1)[0],
        "rt-soa1": normal(0.7, beta_noise, 1)[0],
        "visOnset-soa2": normal(0.8, beta_noise, 1)[0],
        "visOffset-soa2": normal(0.5, beta_noise, 1)[0],
        "aud-soa2": normal(0.4, beta_noise, 1)[0],
        "rt-soa2": normal(0.7, beta_noise, 1)[0],
        "Intercept": normal(0, beta_noise, 1)[0]
    }, index=[0])
    regressor_names = list(true_betas.columns)  # Extract regressor name
    true_beta_values = np.squeeze(np.array(true_betas))  # Get beta values

    # =========================================
    # Create the condition matrices:
    # Generating a counter balanced condition matrix:
    condition_matrix = []
    ctr = 0
    for cond_i in enumerate(CONDITIONS_NAMES):
        vect = np.zeros(N_CONDITIONS * ntrials_per_cond)
        vect[ctr:ctr + ntrials_per_cond] = 1
        condition_matrix.append(vect)
        ctr += ntrials_per_cond
    condition_matrix = np.array(condition_matrix).T

    # =========================================
    # Create the timing matrix:
    # This matrix contains the value of each event in each trial:
    timing_matrix = pd.DataFrame({
        "visOnset": [0] * ntrials_per_cond * N_CONDITIONS,
        "visOffset": [1.5] * ntrials_per_cond * N_CONDITIONS,
        "aud": [0.1] * ntrials_per_cond + [0.3] * ntrials_per_cond,
        "rt": list(normal(0.6, 0.2, ntrials_per_cond * N_CONDITIONS)),
    }).to_numpy()

    # Finally, create the latency matrices:
    # This matrix contains latency values added to the timing matrix. This is the parameter that we actually wish
    # to retrieve.
    true_latencies = np.squeeze(pd.DataFrame({
        "visOnset-soa1": 0,
        "visOffset-soa1": 0,
        "aud-soa1": 0.3,
        "rt-soa1": 0,
        "visOnset-soa2": 0,
        "visOffset-soa2": 0,
        "aud-soa2": 0.2,
        "rt-soa2": 0
    }, index=[0]).to_numpy())

    # =======================================================================
    # Simulate the data:
    design_matrix, times_concat = create_design_matrix(timing_matrix, condition_matrix, true_latencies, purf, TIMES,
                                                       add_jitter=jitter)
    # Plot the design matrix:
    sim_data = np.sum(true_beta_values * design_matrix, axis=1) + np.random.normal(0, noise, [design_matrix.shape[0]])
    plot_glm(sim_data, times_concat, np.sum(true_beta_values * design_matrix, axis=1), true_beta_values * design_matrix,
             regressor_names, axs=None, decorate_axes=True)
    if show_plots:
        plt.show()  # Show the final plot
    else:
        plt.close()

    # =======================================================================
    # Retrieve parameters using glm:
    # Define the range bounds. We can enforce certain parameters not to vary in this way:
    range_bounds = [
        [0, 0, 1],  # Visual onset
        [0, 0, 1],
        [0, 0.5, 100],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0.5, 100],
        [0, 0, 1]
    ]
    optimal_latencies, glm_results, expl_results = latency_glm(sim_data, purf, timing_matrix, condition_matrix, TIMES,
                                                               range_bounds=range_bounds)

    # =======================================================================
    # Plot the results:
    # Plot the optmization results:
    if expl_results.shape[0] > 20:
        steps = int(expl_results.shape[0] / 20)
        expl_results = expl_results.iloc[::steps, :]
    ax = None
    plt.ion()
    for i, row in expl_results.iterrows():
        # Create the design matrix:
        latencies = row[[col for col in row.index if "lat:" in col]].to_numpy()
        expl_betas = row[[col for col in row.index if "beta:" in col]].to_numpy()
        design_matrix_expl, times_concat = create_design_matrix(timing_matrix, condition_matrix,
                                                                latencies, purf, TIMES, verbose=False)
        weighted_design_matrix = expl_betas * design_matrix_expl
        # Generate fitted data:
        fitted_data = np.sum(weighted_design_matrix, axis=1)

        # Plot the data:
        if i == 0:
            ax = plot_glm(sim_data, times_concat, fitted_data, weighted_design_matrix, regressor_names,
                          axs=ax, decorate_axes=True)
        else:
            ax = plot_glm(sim_data, times_concat, fitted_data, weighted_design_matrix, regressor_names,
                          axs=ax, decorate_axes=False, clear_axes=True)
        # Little pause to see it:
        plt.pause(0.3)
        plt.draw()
    plt.ioff()  # Turn off interactive mode after the loop finishes
    plt.close()
    # Plot the best fit:
    obs_betas = np.squeeze(glm_results[[col for col in glm_results.columns if "beta:" in col]].to_numpy())
    design_matrix_best, times_concat = create_design_matrix(timing_matrix, condition_matrix,
                                                            optimal_latencies, purf, TIMES)
    weighted_design_matrix = obs_betas * design_matrix_best
    # Generate fitted data:
    fitted_data = np.sum(weighted_design_matrix, axis=1)
    plot_glm(sim_data, times_concat, fitted_data, weighted_design_matrix, regressor_names,
             axs=None, decorate_axes=True)
    if show_plots:
        plt.show()  # Show the final plot
    else:
        plt.close()

    # Plot the parameters comparisons:
    plot_parameters_comparison(true_beta_values, obs_betas, true_latencies, optimal_latencies, regressor_names, ax=None)
    if show_plots:
        plt.show()  # Show the final plot
    else:
        plt.close()

    # =======================================================================
    # Package the results:
    # Compute simulation performances:
    sim_run_params = pd.DataFrame({
        "ntrials": N_CONDITIONS * ntrials_per_cond,
        "n_conditions": N_CONDITIONS,
        "n_parameters": design_matrix_best.shape[1],
        "jitter": jitter,
        "beta_noise": beta_noise,
        "noise": noise
    }, index=[0])

    # Package the ground truth values in a data frame as well:
    true_parameters_val = list(true_betas) + list(true_latencies)
    keys = (["trueBeta:{}".format(reg) for reg in regressor_names] +
            ["trueLatency:{}".format(reg) for reg in regressor_names[0:-1]])
    true_parameters = pd.DataFrame(
        dict(zip(keys, true_parameters_val)), index=[0]
    )
    # Combine everything into a single data frame:
    sim_run_results = pd.concat([sim_run_params, true_parameters, glm_results], axis=1)

    return sim_run_results


if __name__ == "__main__":
    jitters = [0.001, 0.01, 0.1]
    ntrials_per_conds = [4]
    noises = [1 / 16, 1 / 8, 1 / 4, 1 / 2]
    beta_noises = [1 / 16, 1 / 8, 1 / 4, 1 / 2]
    n_iter = 20
    simulation_results = []
    for jit in jitters:
        for ntrials in ntrials_per_conds:
            for noi in noises:
                for b_noise in beta_noises:
                    for i in range(n_iter):
                        # Set the seed for each iteration:
                        np.random.seed(i)
                        simulation_results.append(run_simulation(jit, ntrials, noi, b_noise, show_plots=False))
    # Save the results:
    simulation_results = pd.concat(simulation_results, axis=0).reset_index(drop=True)
    simulation_results.to_csv("simulation_results.csv")
