import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
import pandas as pd
from statsmodels.regression.linear_model import OLS
import warnings
from tqdm import tqdm
from eye_tracker.pupil_size_analysis.modelling_helper_functions import purf, create_sticks, fitting_function, \
    experiment_model, data_plotter, simulator, estimate_latencies

prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']

# ======================================================================================================================
# Simulation:
# Set Simulation parameters:
sfreq = 500
# Create the times vector:
times = np.arange(-0.2, 3.5, 1 / sfreq)
ntrials = 4
# Create the trial matrix:
trial_matrix = pd.DataFrame({
    "visual_onset": np.zeros(ntrials),
    "auditory_onset": np.array([0.4, 0.6, 0.8, 1.0]),
    "auditory_response": np.array([0.3 + 0.6 if val < 0.3 else val + 0.6
                                   for val in [0.4, 0.6, 0.8, 1.0]]),
    "visual_offset": np.array([1.5] * ntrials),
})

# Create the regressors array and the sticks:
latencies_matrices = pd.DataFrame({
    "latency_vis": np.zeros(ntrials),
    "latency_aud": np.array([0.3 if val < 0.3 else val for val in trial_matrix["auditory_onset"]]),
    "auditory_response": trial_matrix["auditory_response"],
    "visual_offset": trial_matrix["visual_offset"],
})

# Simulate the data:
data = simulator(times, latencies_matrices, tmax=0.93, n=10.1)

# Plot the data:
axs = data_plotter(data, times, latencies_matrices, show_legend=False)
# Remove the xlabel of each plot:
for ax in axs:
    ax.set_xlabel("")
axs[-1].set_xlabel("Time (s)")
plt.show()

# Retrieve the parameters:
estimated_latencies = estimate_latencies(data, times, trial_matrix, [0.5] * trial_matrix.shape[0], n_iterations=100,
                                         n_initial_values=2000, n_best_values=40)

# Plot the results
axs = data_plotter(data, times, latencies_matrices, plot_data=True, plot_timings=True, plot_purf=False,
                   plot_predicted=False)
# Add the distribution of the latencies:
for trial in range(len(trial_matrix)):
    positions = np.linspace(0, 1, len(trial_matrix.columns), endpoint=False)
    for i, evt in enumerate(trial_matrix.columns):
        axs[trial].boxplot(estimated_latencies.loc[estimated_latencies["trial"] == i, evt].to_numpy(),
                           positions=[positions[i]], widths=0.1, vert=False,
                           patch_artist=True, boxprops=dict(facecolor=cwheel[i], color=cwheel[i]))
# Remove the xlabel of each plot:
for ax in axs:
    ax.set_xlabel("")
axs[-1].set_xlabel("Time (s)")
plt.show()
