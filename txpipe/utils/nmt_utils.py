import os
import pymaster as nmt
import numpy as np
import healpy
import pathlib


class WorkspaceCache:
    def __init__(self, dirname, low_mem=False):
        self.path = pathlib.Path(dirname)
        self.path.mkdir(parents=True, exist_ok=True)
        self.keys = {}
        self.low_mem = low_mem
        self.workspaces = {}

    def get(self, i, j, k, key=None):
        workspace = self.workspaces.get((i, j, k))
        if workspace is not None:
            return workspace

        if key is None:
            try:
                key = self.keys[i, j, k]
            except KeyError:
                raise KeyError("Tried to get workspace neither already cached nor with key given")

        p = self.get_path(key)

        if not p.exists():
            return None

        # Initialize a workspace and populate
        # it from file
        workspace = nmt.NmtWorkspace()
        workspace.read_from(str(p))
        workspace.txpipe_key = key

        if not self.low_mem:
            self.workspaces[i, j, k] = workspace

        return workspace

    def get_path(self, key):
        return self.path / f"workspace_{key}.dat"

    def put(self, i, j, k, workspace):
        key = workspace.txpipe_key
        self.keys[i, j, k] = key
        p = self.get_path(key)

        if not self.low_mem:
            self.workspaces[i, j, k] = workspace

        if p.exists():
            return

        else:
            os.makedirs(os.path.dirname(str(p)), exist_ok=True)

        workspace.write_to(str(p))


def choose_ell_bins(**config):
    """Create an NmtBin object based on configuration parameters,
    choosing an ell binning scheme.

    These three methods are attempted, in this order:
    1. If 'ell_edges' is provided, use those edges directly.
    2. If 'ell_min', 'ell_max', and 'n_ell' are provided, use those to
       create evenly spaced bins, in log-space by default, or linear
       space if 'ell_spacing' is set to 'linear' or 'lin'.

    3. If 'bandpower_width' and 'nside' are provided, use those to
         create bins of the given width, using nmt.NmtBin.from_nside_linear.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing one of the above sets of parameters.

    Returns
    -------
    ell_bins : nmt.NmtBin
        The resulting NmtBin object.
    """
    from pymaster import NmtBin

    if "ell_edges" in config:
        ell_edges = config["ell_edges"]
        ell_bins = NmtBin.from_edges(ell_edges)

    elif "ell_min" in config:
        if not "ell_max" in config or not "n_ell" in config:
            raise ValueError("When specifying ell_min for ell binning, must also specify ell_max and n_ell")
        ell_min = config["ell_min"]
        ell_max = config["ell_max"]
        n_ell = config["n_ell"]
        ell_spacing = config.get("ell_spacing", "log")

        if ell_spacing == "linear" or ell_spacing == "lin":
            ell_edges = np.unique(np.linspace(ell_min, ell_max, n_ell).astype(int))
        elif ell_spacing == "log":
            ell_edges = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))

        ell_bins = NmtBin.from_edges(ell_edges[:-1], ell_edges[1:])

    elif "bandpower_width" in config:
        if not "nside" in config:
            raise ValueError("When specifying bandpower_width for ell binning, must also specify nside")
        bandpower_width = config["bandpower_width"]
        nside = config["nside"]
        ell_bins = nmt.NmtBin.from_nside_linear(nside, nlb=bandpower_width)

    else:
        raise ValueError(
            "No valid ell binning configuration found. Specify one of ell_edges, ell_min/ell_max/n_ell, or bandpower_width/nside."
        )

    return ell_bins
