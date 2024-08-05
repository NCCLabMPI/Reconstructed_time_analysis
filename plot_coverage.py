import pickle
from os import listdir
import environment_variables as ev
from pathlib import Path
import mne
from helper_function.helper_plotter import plot_rois, get_color_mapping
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# Directory of the results:
save_dir = Path(ev.bids_root, "derivatives", "decoding_10ms", "Dur", "decoding_auc")
# Set list of views:
views = {'side': {"azimuth": 180, "elevation": 90}, 'front': {"azimuth": 130, "elevation": 90},
         "ventral": {"azimuth": 90, "elevation": 180}}

# Prepare a dict for the results of each ROI:
roi_cts = {}

# Loop through all pickle files:
for file in [path for path in listdir(save_dir) if path.endswith('pkl')]:
    if "all_roi" in file:
        continue
    with open(Path(save_dir, file), "rb") as fp:
        res = pickle.load(fp)
    roi_name = file.split("results-")[1].split(".pkl")[0]
    roi_cts[roi_name] = res['n_channels']

# Add the ROIs which are missing:
labels = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', hemi='both', surf_name='pial',
                                    subjects_dir=ev.fs_directory, sort=True)
roi_names = list(set(lbl.name.replace("-lh", "").replace("-rh", "")
                     for lbl in labels if "unknown" not in lbl.name.lower()))


# Now, seprate the roi where we have coverage from those where we don't:
covered_rois = {roi: roi_cts[roi] for roi in roi_cts.keys() if roi_cts[roi] >= 10}
non_covered_rois = {roi: [1, 1, 1] for roi in roi_cts.keys() if roi_cts[roi] < 10}
non_covered_rois.update({roi: [1, 1, 1] for roi in roi_names if roi not in roi_cts.keys()})

# Get the ROIs colors:
rois_colors = get_color_mapping(covered_rois, color_map="cividis", max_prctile=0.2)
rois_colors.update(non_covered_rois)
brain = plot_rois(ev.fs_directory, "fsaverage", "aparc.a2009s", rois_colors, plot_borders=False)
for view in views:
    brain.show_view(**views[view])
    brain.save_image(Path(save_dir, "{}_{}.png".format("coverage", view)))
brain.close()
# Plot a colorbar:
min_val = min(covered_rois.values())
max_val = max(covered_rois.values())
norm = mpl.colors.Normalize(vmin=min_val,
                            vmax=max_val - (max_val - min_val) * 0.2)
fig, ax = plt.subplots(figsize=(1, 6))
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='cividis'),
             cax=ax, orientation='vertical', label='Duration (s)')
fig.savefig(Path(save_dir, "colorbar_coverage.svg"),
            transparent=False, dpi=500)
plt.close()

fig, ax = plt.subplots(figsize=(6, 1))
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='cividis'),
             cax=ax, orientation='horizontal', label='Duration (s)')
fig.savefig(Path(save_dir, "colorbar_coverage_horz.svg"),
            transparent=False, dpi=500)
plt.close()


# Print the names of the ROIs with no coverage:
for roi in non_covered_rois.keys():
    print(roi)
