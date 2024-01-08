import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eye_tracker.pupil_size_analysis.pupil_size_helper_functions import (purf_fun, create_design_matrix, latency_glm,
                                                                         plot_glm, plot_parameters_comparison,
                                                                         create_stick_predictors)

prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']

# Set fixed parameters:
SFREQ = 100
tmin = -0.2
tmax = 4.0
TIMES = np.arange(tmin, tmax, 1 / SFREQ)
PURF_TMAX = 0.93
PURF_N = 10.1
regressor_names = ["Onset", "Audio", "Offset"]
# Create the purf:
purf = purf_fun(TIMES[TIMES > 0], PURF_TMAX, n=PURF_N)
betas = np.array([0.6, 0.1, 0.2])
latencies = [0, 0.1, 1.5]
# Create stick predictors:
sticks_predictors, times_concat = create_stick_predictors(pd.DataFrame(np.array([latencies])), TIMES)
# Create the design matrix:
design_matrix = np.array([np.convolve(sticks_predictors[:, i], purf, mode='full')[0:sticks_predictors.shape[0]]
                          for i in range(sticks_predictors.shape[1])]).T
# GLM:
sim_data = np.sum(betas * design_matrix, axis=1) + np.random.normal(0, 0.2, [design_matrix.shape[0]])

# Plot the data:
fig, axs = plt.subplots(2, 1, layout='constrained')
# Plot the observed data:
axs[0].plot(TIMES, sim_data, label="Data", color="blue")
# Plot the fitted data:
axs[0].plot(TIMES, np.sum(betas * design_matrix, axis=1), label="Fitted", color="red")
axs[0].legend()
# Plot each component separately:
[axs[1].plot(TIMES, design_matrix[:, i], label=regressor_names[i], color=cwheel[i])
 for i in range(design_matrix.shape[1])]
# Plot each stick:
[axs[1].vlines(latencies[i], 0, 1, linestyles='dashed', color=cwheel[i])
 for i in range(len(latencies))]
axs[1].legend()

# Set labels:
axs[0].set_ylabel("Pupil size")
axs[1].set_xlabel("Time (sec)")
axs[0].set_title("Fitted data")
axs[0].spines[['right', 'top']].set_visible(False)
axs[1].set_title("Design matrix")
axs[1].spines[['right', 'top']].set_visible(False)
plt.show()
