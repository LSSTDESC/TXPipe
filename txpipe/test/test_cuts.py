from ..source_selection import TXSimpleSelection, TXMetaDetectSelection, TXScalarMetaDetectSelection, TXMetacalSelection, TXLensfitSelection, TXHSCSelection
from ..tomography import TXSimpleRedshiftSourceTomography
from ..shear_calibration.simple import TXComputeShearCalibrationSimple
from ..shear_calibration.metadetect import TXComputeShearCalibrationMetadetect
from ..shear_calibration.scalar_metadetect import TXComputeShearCalibrationScalarMetadetect
from ..shear_calibration.metacal import TXComputeShearCalibrationMetacal
from ..shear_calibration.lensfit import TXComputeShearCalibrationLensfit
from ..calibration_utils.names import META_VARIANTS, SCALAR_META_VARIANTS, metacal_variants, metadetect_variants, scalar_metadetect_variants
from tempfile import TemporaryDirectory
import os
import h5py
import numpy as np


def make_config(dirname):
    """
    Make an empty config file in the directory
    """
    filename = os.path.join(dirname, "config.yml")
    with open(filename, "w") as f:
        f.write("global: {}")
    return filename

def make_simple_mock_shear_catalog(dirname):
    filename = os.path.join(dirname, "shear_catalog.hdf5")
    input_filename = "data/example/inputs/metadetect_shear_catalog.hdf5"
    fin = h5py.File(input_filename, "r")
    gin = fin["shear/ns"]
    with h5py.File(filename, "w") as f:
        g = f.create_group("shear")
        g.attrs["catalog_type"] = "simple"
        for k in gin.keys():
            g.create_dataset(k, data=gin[k][:])
    fin.close()
    return filename


def assert_populated_bins(values, nbin=3):
    counts = np.unique_counts(values)
    print(counts)
    for i in range(nbin):
        assert i in counts.values
        j = np.where(counts.values == i)[0]
        assert counts.counts[j] > 0



def test_simple_cuts():
    with TemporaryDirectory() as dirname:
        config = make_config(dirname)
        shear_cat = make_simple_mock_shear_catalog(dirname)
        shear_sel = os.path.join(dirname, "shear_selection.hdf5")
        shear_bin = os.path.join(dirname, "shear_bin_assignments.hdf5")
        tomo_cat = os.path.join(dirname, "shear_tomography_catalog.hdf5")

        args = {
            # file paths
            "shear_catalog": shear_cat,
            "shear_selection": shear_sel,
            "config": config,
            # and some configuration options
            "s2n_min": 10,
            "T_min": 0.5,
            "verbose": True,
        }

        stage = TXSimpleSelection(args)
        stage.run()
        stage.finalize()
        assert os.path.exists(shear_sel)

        with h5py.File(shear_sel) as f:
            sel = f["shear/selection"][:]
            print(sel.sum() / sel.size)

        # Now do some tomography
        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "config": config,
            "source_zbin_edges": [0.1, 0.2, 0.3, 0.4],
            "redshift_col": "redshift_true",
            "chunk_rows": 10_000,
        }
        stage = TXSimpleRedshiftSourceTomography(args)
        stage.run()
        stage.finalize()


        with h5py.File(shear_bin) as f:
            sel = f[f"bin"][:]
            print(sel.sum())
            assert_populated_bins(sel)


        # finally combine these two into a full summary catalog
        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "shear_selection": shear_sel,
            "shear_tomography_catalog": tomo_cat,
            "config": config,
            "chunk_rows": 10_000,
            "verbose": True,
        }
        stage = TXComputeShearCalibrationSimple(args)
        stage.run()
        stage.finalize()

        with h5py.File(tomo_cat) as f:
            counts = f['counts/counts'][:]
            assert np.all(counts > 0)
            tomo = f['tomography/bin'][:]
            assert_populated_bins(tomo)



def test_metadetect_cuts():
    with TemporaryDirectory() as dirname:
        config = make_config(dirname, )
        shear_cat = "data/example/inputs/metadetect_shear_catalog.hdf5"
        shear_sel = os.path.join(dirname, "shear_selection.hdf5")
        shear_bin = os.path.join(dirname, "shear_bin_assignments.hdf5")
        tomo_cat = os.path.join(dirname, "shear_tomography_catalog.hdf5")

        args = {
            # file paths
            "shear_catalog": shear_cat,
            "shear_selection": shear_sel,
            "config": config,
            # and some configuration options
            "s2n_min": 10,
            "T_min": 0.5,
            "verbose": True,
        }
        stage = TXMetaDetectSelection(args)
        stage.run()
        stage.finalize()
        assert os.path.exists(shear_sel)

        with h5py.File(shear_sel) as f:
            for v in META_VARIANTS:
                sel = f[f"shear/{v}/selection"][:]
                assert sel.sum() > 0
                print(v, sel.sum() / sel.size)

        # Now do some tomography
        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "config": config,
            "source_zbin_edges": [0.1, 0.2, 0.3, 0.4],
            "redshift_col": "redshift_true",
            "chunk_rows": 10_000,
        }
        stage = TXSimpleRedshiftSourceTomography(args)
        stage.run()
        stage.finalize()

        with h5py.File(shear_bin) as f:
            for col in metadetect_variants("bin"):
                sel = f[col][:]
                print(col)
                assert_populated_bins(sel)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "shear_selection": shear_sel,
            "shear_tomography_catalog": tomo_cat,
            "config": config,
            "bands": ["r", "i", "z"],
            "chunk_rows": 10_000,
            "delta_gamma": 0.02,
            "true_z": True,
            "input_pz": False,
            "verbose": True,
        }
        stage = TXComputeShearCalibrationMetadetect(args)
        stage.run()
        stage.finalize()

        with h5py.File(tomo_cat) as f:
            counts = f["counts/counts"][:]
            assert np.all(counts > 0)
            for col in metadetect_variants("bin"):
                tomo = f[f"tomography/{col}"][:]
                assert_populated_bins(tomo)
        




def test_scalar_metadetect_cuts():
    with TemporaryDirectory() as dirname:
        config = make_config(dirname)
        shear_cat = "data/example/inputs/scalar_metadetect_shear_catalog.hdf5"
        shear_sel = os.path.join(dirname, "shear_selection.hdf5")
        shear_bin = os.path.join(dirname, "shear_bin_assignments.hdf5")
        tomo_cat = os.path.join(dirname, "shear_tomography_catalog.hdf5")

        args = {
            # file paths
            "shear_catalog": shear_cat,
            "shear_selection": shear_sel,
            "config": config,
            # and some configuration options
            "s2n_min": 10,
            "T_min": 0.5,
            "verbose": True,
        }
        stage = TXScalarMetaDetectSelection(args)
        stage.run()
        stage.finalize()
        assert os.path.exists(shear_sel)

        with h5py.File(shear_sel) as f:
            for col in scalar_metadetect_variants("selection"):
                sel = f[f"shear/{col}"][:]
                assert sel.sum() > 0
                print(col, sel.sum() / sel.size)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "config": config,
            "source_zbin_edges": [0.1, 0.2, 0.3, 0.4],
            "redshift_col": "redshift_true",
            "chunk_rows": 10_000,
        }
        stage = TXSimpleRedshiftSourceTomography(args)
        stage.run()
        stage.finalize()

        with h5py.File(shear_bin) as f:
            for col in scalar_metadetect_variants("bin"):
                sel = f[col][:]
                print(col)
                assert_populated_bins(sel)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "shear_selection": shear_sel,
            "shear_tomography_catalog": tomo_cat,
            "config": config,
            "bands": ["r", "i", "z"],
            "chunk_rows": 10_000,
            "delta_gamma": 0.02,
            "true_z": True,
            "input_pz": False,
            "verbose": True,
        }
        stage = TXComputeShearCalibrationScalarMetadetect(args)
        stage.run()
        stage.finalize()

        with h5py.File(tomo_cat) as f:
            counts = f["counts/counts"][:]
            assert np.all(counts > 0)
            for col in scalar_metadetect_variants("bin"):
                tomo = f[f"tomography/{col}"][:]
                assert_populated_bins(tomo)



def test_metacal_cuts():
    with TemporaryDirectory() as dirname:
        config = make_config(dirname)
        shear_cat = "data/example/inputs/shear_catalog.hdf5"
        shear_sel = os.path.join(dirname, "shear_selection.hdf5")
        shear_bin = os.path.join(dirname, "shear_bin_assignments.hdf5")
        tomo_cat = os.path.join(dirname, "shear_tomography_catalog.hdf5")

        args = {
            # file paths
            "shear_catalog": shear_cat,
            "shear_selection": shear_sel,
            "config": config,
            # and some configuration options
            "s2n_min": 10,
            "T_min": 0.5,
            "verbose": True,
        }
        stage = TXMetacalSelection(args)
        stage.run()
        stage.finalize()
        assert os.path.exists(shear_sel)

        with h5py.File(shear_sel) as f:
            print(f['shear'].keys())
            for col in metacal_variants("selection"):
                sel = f[f"shear/{col}"][:]                
                assert sel.sum() > 0
                print(col, sel.sum() / sel.size)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "config": config,
            "source_zbin_edges": [0.1, 0.2, 0.3, 0.4],
            "redshift_col": "redshift_true",
            "chunk_rows": 10_000,
        }
        stage = TXSimpleRedshiftSourceTomography(args)
        stage.run()
        stage.finalize()

        with h5py.File(shear_bin) as f:
            # check the baseline no-shear version.
            for col in metacal_variants("bin"):
                sel = f[col][:]
                assert_populated_bins(sel)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "shear_selection": shear_sel,
            "shear_tomography_catalog": tomo_cat,
            "config": config,
            "bands": ["r", "i", "z"],
            "chunk_rows": 10_000,
            "delta_gamma": 0.02,
            "true_z": True,
            "input_pz": False,
            "use_diagonal_response": False,
            "verbose": True,
        }
        stage = TXComputeShearCalibrationMetacal(args)
        stage.run()
        stage.finalize()

        with h5py.File(tomo_cat) as f:
            counts = f["counts/counts"][:]
            assert np.all(counts > 0)
            tomo = f["tomography/bin"][:]
            assert_populated_bins(tomo)


def test_lensfit_cuts():
    with TemporaryDirectory() as dirname:
        config = make_config(dirname)
        shear_cat = "data/example/inputs/lensfit_shear_catalog.hdf5"
        shear_sel = os.path.join(dirname, "shear_selection.hdf5")
        shear_bin = os.path.join(dirname, "shear_bin_assignments.hdf5")
        tomo_cat = os.path.join(dirname, "shear_tomography_catalog.hdf5")

        args = {
            "shear_catalog": shear_cat,
            "shear_selection": shear_sel,
            "config": config,
            "s2n_min": 10,
            "T_min": 0.5,
            "verbose": True,
        }
        stage = TXLensfitSelection(args)
        stage.run()
        stage.finalize()
        assert os.path.exists(shear_sel)

        with h5py.File(shear_sel) as f:
            sel = f["shear/selection"][:]
            assert sel.sum() > 0
            print(sel.sum() / sel.size)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "config": config,
            "source_zbin_edges": [0.1, 0.2, 0.3, 0.4],
            "redshift_col": "mean_z",
            "chunk_rows": 10_000,
        }
        stage = TXSimpleRedshiftSourceTomography(args)
        stage.run()
        stage.finalize()

        with h5py.File(shear_bin) as f:
            sel = f["bin"][:]
            assert_populated_bins(sel)

        args = {
            "shear_catalog": shear_cat,
            "shear_bin_assignments": shear_bin,
            "shear_selection": shear_sel,
            "shear_tomography_catalog": tomo_cat,
            "config": config,
            "chunk_rows": 10_000,
            "input_m_is_weighted": True,
            "verbose": True,
        }
        stage = TXComputeShearCalibrationLensfit(args)
        stage.run()
        stage.finalize()

        with h5py.File(tomo_cat) as f:
            counts = f["counts/counts"][:]
            assert np.all(counts > 0)
            tomo = f["tomography/bin"][:]
            assert_populated_bins(tomo)


