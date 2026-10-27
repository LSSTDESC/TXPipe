"""
Tiny fake input files for the cluster_counts stage tests.

The sources are put on rings around the clusters, with a pure tangential
shear, so that the expected outputs of each stage are known exactly.
"""
import os
import numpy as np
import h5py

# the tests are run from the repository root, like the other TXPipe tests
FIDUCIAL_COSMOLOGY = "data/fiducial_cosmology.yml"

# redshift grid and width of the fake photo-z PDFs
PDF_Z = np.linspace(0.0, 3.0, 301)
PDF_SIGMA = 0.02

META_VARIANTS = ["ns", "1p", "1m", "2p", "2m"]


def write_cluster_catalog(filename, ra, dec, redshift, richness):
    n = len(ra)
    with h5py.File(filename, "w") as f:
        g = f.create_group("clusters")
        g.create_dataset("cluster_id", data=np.arange(n, dtype=np.int64))
        g.create_dataset("ra", data=np.array(ra, dtype=float))
        g.create_dataset("dec", data=np.array(dec, dtype=float))
        g.create_dataset("redshift", data=np.array(redshift, dtype=float))
        g.create_dataset("redshift_err", data=np.zeros(n))
        g.create_dataset("richness", data=np.array(richness, dtype=float))
        g.create_dataset("richness_err", data=np.zeros(n))
        g.create_dataset("scaleval", data=np.ones(n))


def ring(ra_cluster, dec_cluster, radius_arcmin, n):
    """
    Positions of n objects on a ring around a cluster, and their
    position angle phi measured from the RA direction.
    """
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius_arcmin / 60.0
    ra = ra_cluster + r * np.cos(phi) / np.cos(np.radians(dec_cluster))
    dec = dec_cluster + r * np.sin(phi)
    return ra, dec, phi


def tangential_shear(phi, g_t):
    """
    g1, g2 giving a tangential shear g_t and no cross shear,
    with the CLMM sign convention.
    """
    return -g_t * np.cos(2 * phi), -g_t * np.sin(2 * phi)


def write_source_catalogs(dirname, ra, dec, g1, g2, zmode, zmean, ztrue, zpdf, source_bin,
                          catalog_type="simple", R=None):
    """
    Write a shear catalog, the matching shear tomography catalog and
    a qp-style photo-z file, and return a dict of their paths.

    g1 and g2 are the true shears. For metadetect the catalog stores
    R @ g, so that the calibrated shear is g again.
    """
    n = len(ra)
    paths = {
        "shear_catalog": os.path.join(dirname, "shear_catalog.hdf5"),
        "shear_tomography_catalog": os.path.join(dirname, "shear_tomography_catalog.hdf5"),
        "source_photoz_pdfs": os.path.join(dirname, "source_photoz_pdfs.hdf5"),
    }

    if R is None:
        R = np.eye(2)
    g_obs = R @ np.array([g1, g2])
    cols = {
        "ra": ra,
        "dec": dec,
        "g1": g_obs[0],
        "g2": g_obs[1],
        "weight": np.ones(n),
        "redshift_true": ztrue,
    }

    with h5py.File(paths["shear_catalog"], "w") as f:
        g = f.create_group("shear")
        g.attrs["catalog_type"] = catalog_type
        if catalog_type == "simple":
            for name, col in cols.items():
                g.create_dataset(name, data=col)
        elif catalog_type == "metadetect":
            for v in META_VARIANTS:
                for name, col in cols.items():
                    g.create_dataset(f"{v}/{name}", data=col)
        else:
            raise ValueError(f"No fake shear catalog for {catalog_type}")

    with h5py.File(paths["shear_tomography_catalog"], "w") as f:
        g = f.create_group("tomography")
        g.attrs["catalog_type"] = catalog_type
        g.attrs["nbin"] = 1
        g.create_dataset("bin", data=np.array(source_bin, dtype=np.int32))
        c = f.create_group("counts")
        for name in ["mean_e1", "mean_e2", "mean_e1_2d", "mean_e2_2d"]:
            c.create_dataset(name, data=np.zeros(1))
        if catalog_type == "metadetect":
            r = f.create_group("response")
            r.create_dataset("R", data=R[np.newaxis])
            r.create_dataset("R_2d", data=R)

    # Gaussian PDFs centred on zpdf, in the qp layout
    pdfs = np.exp(-0.5 * ((PDF_Z[np.newaxis] - np.array(zpdf)[:, np.newaxis]) / PDF_SIGMA) ** 2)
    with h5py.File(paths["source_photoz_pdfs"], "w") as f:
        f.create_dataset("ancil/zmode", data=np.array(zmode, dtype=float))
        f.create_dataset("ancil/zmean", data=np.array(zmean, dtype=float))
        f.create_dataset("data/yvals", data=pdfs)
        f.create_dataset("meta/xvals", data=PDF_Z[np.newaxis])

    return paths


def run_stage(cls, **args):
    """
    Build a stage from its input/output paths and config options,
    run it, and move its outputs to their final names.
    """
    args.setdefault("config", {})
    stage = cls(args)
    stage.run()
    stage.finalize()
    return stage
