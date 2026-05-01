from ..mapping.basic_maps import pix2sparseindex
import numpy as np
import healsparse as hsp
import healpy as hp

def _make_hsp_map(nside_sparse=4096, nside_coverage=32):
    """
    Build a HealSparseMap and set a handful of pixels 
    so that a few coverage blocks are active.
    """
    m = hsp.HealSparseMap.make_empty(
        nside_coverage=nside_coverage,
        nside_sparse=nside_sparse,
        dtype=np.float32,
    )

    # Touch pixels in three distinct coverage blocks (0, 1, 5)
    nfine = m._cov_map.nfine_per_cov
    pixels_to_set = np.array([
        0,               # coverage block 0
        nfine,           # coverage block 1
        5 * nfine,       # coverage block 5
    ], dtype=np.int64)

    #set to random values
    np.random.seed(1)
    m.update_values_pix(pixels_to_set, np.random.rand(len(pixels_to_set)).astype(np.float32) )
    return m


# ---------------------------------------------------------------------------
# test sparse_index computation
# ---------------------------------------------------------------------------

def test_pix2sparseindex_single_valid_pixel():
    hsp_map = _make_hsp_map()
    cov_map = hsp_map._cov_map
    
    pix_val = np.int64(0)
    assert pix_val in hsp_map.valid_pixels
    
    pix = np.array([pix_val])

    sparse_index, sparse_npix = pix2sparseindex(pix, cov_map, return_npix=True)
    result = np.asarray(sparse_index.compute())

    # compute expected result using numpy 
    bit_shift = cov_map._bit_shift
    cov_pix = np.right_shift(pix, bit_shift)
    expected_result = cov_map._cov_index_map[cov_pix] + pix

    # check the values match
    assert (hsp_map._sparse_map[result] == hsp_map[pix]).all()

    # also check the npix is correct
    assert sparse_npix == hsp_map._sparse_map.shape[0]

    # also check index matches the numpy result
    assert (result == expected_result).all()

    # also check the sparse index is not in the empty pixel 
    # (since we know this is a valid pixel)
    assert (result >= cov_map.nfine_per_cov).all()

def test_pix2sparseindex_single_invalid_pixel():
    hsp_map = _make_hsp_map()
    cov_map = hsp_map._cov_map
    
    pix_val = np.int64(114688+1)
    assert pix_val not in hsp_map.valid_pixels

    pix = np.array([pix_val])

    sparse_index, sparse_npix = pix2sparseindex(pix, cov_map, return_npix=True)
    result = np.asarray(sparse_index.compute())

    # compute expected result using numpy 
    bit_shift = cov_map._bit_shift
    cov_pix = np.right_shift(pix, bit_shift)
    expected_result = cov_map._cov_index_map[cov_pix] + pix

    #check the values match
    assert (hsp_map._sparse_map[result] == hsp_map[pix]).all()

    # also check the npix is correct
    assert sparse_npix == hsp_map._sparse_map.shape[0]

    # also check index matches the numpy result
    assert (result == expected_result).all()

    # also check the sparse index *is* in the empty pixel 
    # (since we know this is *not* a valid pixel)
    assert (result < cov_map.nfine_per_cov).all()
    assert (hsp_map[result] == hsp_map.sentinel).all()

def test_pix2sparseindex_all_pixels():
    #make a very small map since we are testing all pixels
    hsp_map = _make_hsp_map(nside_sparse=64, nside_coverage=32)
    cov_map = hsp_map._cov_map
    
    npix = hp.nside2npix(hsp_map.nside_sparse)
    pix = np.arange(npix)

    sparse_index, sparse_npix = pix2sparseindex(pix, cov_map, return_npix=True)
    result = np.asarray(sparse_index.compute())

    #check the values of all the pixels match
    assert (hsp_map[pix] == hsp_map._sparse_map[result]).all()