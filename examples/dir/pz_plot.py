#!/usr/bin/env python3
import sys
sys.path.append('.')
import txpipe
import matplotlib.pyplot as plt


with txpipe.data_types.NOfZFile("data/example/dir/lens_photoz_stack.hdf5", "r") as f:
    f.plot('lens')
    plt.legend(frameon=False)
    plt.title("Lens n(z)")
    plt.xlim(xmin=0)


plt.savefig("data/example/dir/plot.png")