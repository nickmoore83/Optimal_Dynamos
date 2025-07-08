"""
Plot scalars.

Usage:
    plot_scalars.py <file>

"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
from docopt import docopt

# Parameters
tasks = ['M', 'E']
figsize = (6, 4)

# Plot
fig = plt.figure(figsize=figsize)
args = docopt(__doc__)
with h5py.File(args['<file>'], mode='r') as file:
    t = np.array(file['scales/sim_time'])
    for task in tasks:
        dset = file['tasks'][task]
        plt.plot(t, dset[:].ravel(), label=task)
plt.xlabel('t')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('scalars_linear.pdf')
plt.yscale('log')
plt.tight_layout()
plt.savefig('scalars_log.pdf')

