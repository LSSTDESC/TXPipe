import numpy as np
import healsparse as hsp
from ..utils import iter_valid_pixels, valid_pixel_sum
from ..data_types import MapsFile


def make_frac_map(nside_coverage=8, nside_sparse=256, seed=1):
    """
    Make a float map with valid pixels spread over several coverage pixels
    """
    rng = np.random.default_rng(seed)
    m = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.float64)
    npix = 12 * nside_sparse**2
    pix = np.unique(rng.integers(0, npix, size=5000))
    m.update_values_pix(pix, rng.uniform(0.01, 1.0, size=pix.size))
    return m


def test_iter_valid_pixels():
    m = make_frac_map()
    pix_chunks = []
    val_chunks = []
    for pix, values in iter_valid_pixels(m):
        assert pix.size > 0
        assert pix.size == values.size
        pix_chunks.append(pix)
        val_chunks.append(values)

    # more than one coverage pixel should have been visited
    assert len(pix_chunks) > 1

    pix = np.concatenate(pix_chunks)
    values = np.concatenate(val_chunks)
    order = np.argsort(pix)
    expected_pix = np.sort(m.valid_pixels)
    np.testing.assert_array_equal(pix[order], expected_pix)
    np.testing.assert_array_equal(values[order], m[expected_pix])


def test_valid_pixel_sum():
    m = make_frac_map()
    np.testing.assert_allclose(valid_pixel_sum(m), np.sum(m[m.valid_pixels]))

    empty = hsp.HealSparseMap.make_empty(8, 256, np.float64)
    assert valid_pixel_sum(empty) == 0.0


def test_read_mask_thresh_and_bool(tmp_path):
    m = make_frac_map()
    filename = str(tmp_path / "mask.hdf5")
    metadata = {"pixelization": "healpix", "nside": m.nside_sparse, "nest": True}

    f = MapsFile(filename, "w")
    f.file.create_group("maps")
    f.write_map("mask", m, metadata)
    f.close()

    thresh = 0.5
    all_pix = m.valid_pixels
    all_vals = m[all_pix]
    expected_pix = np.sort(all_pix[all_vals > thresh])

    f = MapsFile(filename, "r")
    mask = f.read_mask(thresh=thresh)
    bool_mask = f.read_mask(thresh=thresh, returnbool=True)
    f.close()

    np.testing.assert_array_equal(np.sort(mask.valid_pixels), expected_pix)
    np.testing.assert_array_equal(mask[expected_pix], m[expected_pix])

    assert np.issubdtype(bool_mask.dtype, np.bool_)
    np.testing.assert_array_equal(np.sort(bool_mask.valid_pixels), expected_pix)
    assert np.all(bool_mask[expected_pix])
