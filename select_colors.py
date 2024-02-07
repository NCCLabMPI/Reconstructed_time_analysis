from colorspacious import cspace_converter

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl


colors_maps = ["Greens", "Greys"]
conditions = ["T1", "Offset"]
n_colors = 4
soas = [0, 0.116, 0.232, 0.466]

for ci, cmap in enumerate(colors_maps):
    print("Colors for {}".format(conditions[ci]))
    pltcmap = plt.get_cmap(cmap)
    norm = plt.Normalize(0, n_colors)
    cmap_values = pltcmap(norm(np.linspace(1, 4, n_colors)))
    for i, soa in enumerate(soas):
        print("SOA: {}".format(soa))
        print((cmap_values[len(soas) - (i + 1), :3]))




