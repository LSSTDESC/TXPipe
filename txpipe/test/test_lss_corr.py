from ..lsstools import (
    DensityCorrelation,
    linear_model_from_maps,
)
import numpy as np
import healsparse as hsp
import numpy.testing

def test_linear_model():
    """
    We implement a simple linear systematic model in two places
    lsstools.DensityCorrelation.linear_model() (fast, used for fitting)
    lsstools.linear_model_from_maps() (slow, used for making maps of the output model)
    Here we test they give the same answers
    """
    nside_sparse = 128
    nvalid = 1000
    pixels = np.arange(100,100+nvalid)

    #make 3 fake sp maps (uniform number between 0 and 1)
    s1 = hsp.HealSparseMap.make_empty(32, nside_sparse, dtype=float)
    s1.update_values_pix(pixels, np.random.rand(nvalid))
    s2 = hsp.HealSparseMap.make_empty_like(s1)
    s2.update_values_pix(pixels, np.random.rand(nvalid))
    s3 = hsp.HealSparseMap.make_empty_like(s1)
    s3.update_values_pix(pixels, np.random.rand(nvalid))
    sysmaps = [s1,s2,s3]
    sysmap_table = np.vstack([s1[pixels],s2[pixels],s3[pixels]])

    #also make a fake frac map
    frac_map = hsp.HealSparseMap.make_empty_like(s1)
    frac_map.update_values_pix(pixels, np.random.rand(nvalid))
    frac = frac_map[pixels]

    #make a fake catalog 
    nobj = 10000
    pix_obj = np.random.choice(pixels, size=nobj, replace=True)
    s1_obj = s1[pix_obj]
    s2_obj = s2[pix_obj]
    s3_obj = s3[pix_obj]

    #make density correlation object and fill in the 1d correlations
    ds = DensityCorrelation()

    #s1
    ds.add_correlation(
        map_index=0, 
        edges=np.linspace(0,1,11),
        sys_vals=sysmap_table[0],
        data=s1_obj,
        map_input=False, 
        frac=frac, 
        weight=None,
        sys_name="s1",
        use_precompute=False,
        do_grid_hist=False,
        normalize=True,
        )
    ds.add_correlation(
        map_index=1, 
        edges=np.linspace(0,1,11),
        sys_vals=sysmap_table[1],
        data=s2_obj,
        map_input=False, 
        frac=frac, 
        weight=None,
        sys_name="s2",
        use_precompute=False,
        do_grid_hist=False,
        normalize=True,
        )
    ds.add_correlation(
        map_index=2, 
        edges=np.linspace(0,1,11),
        sys_vals=sysmap_table[2],
        data=s3_obj,
        map_input=False, 
        frac=frac, 
        weight=None,
        sys_name="s3",
        use_precompute=False,
        do_grid_hist=False,
        normalize=True,
        )
    
    #These are the maps that were flagges as "significant" (by some separate process)
    maps2correct = np.array([0,2])
    
    #example fit params
    beta = 1.0
    alpha1 = 0.1
    alpha2 = 0.2
    alphas = np.array([alpha1, alpha2])
    params = np.array([beta, alpha1, alpha2])
    
    #run lsstools.DensityCorrelation.linear_model
    A = ds.precompute_design_matrix(sysmap_table, frac=frac, corr_map_indices=maps2correct)
    model_ndens_matrix = ds.linear_model(params)

    #run lsstools.linear_model_from_maps
    corrected_maps = np.array([0,1])
    model_map, model_ds = linear_model_from_maps(
        beta,
        *alphas,
        density_correlation=ds,
        sys_maps=sysmaps,
        sysmap_table=sysmap_table,
        map_index=maps2correct,
        frac=frac_map, 
        do_grid_hist=True,
    )
    model_ndens_map = model_ds.ndens

    numpy.testing.assert_allclose(model_ndens_matrix, model_ndens_map)

