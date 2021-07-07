import numpy as np
from ..base_stage import PipelineStage
from ..data_types import HDFFile, TextFile, SACCFile
import re
import time
import traceback
from ..twopoint import TXTwoPoint, POS_POS




class CMCorrelations(TXTwoPoint):
    name = "CMCorrelations"

    inputs = [
         ("cluster_mag_halo_tomography", HDFFile),
         ("cluster_mag_background", HDFFile),
         ("patch_centers", TextFile),
         ("random_cats", HDFFile),
    ]
    outputs = [
        ('twopoint_data_real_raw', SACCFile),
    ]

    config_options = {
        'calcs': [POS_POS],
        'min_sep': 0.5,
        'max_sep': 300.,
        'nbins': 9,
        'bin_slop': 0.0,
        'sep_units': 'arcmin',
        'cores_per_task': 32,
        'verbose': 1,
        'source_bins': [-1],
        'lens_bins': [-1],
        'reduce_randoms_size': 1.0,
        'do_shear_shear': False,
        'do_shear_pos': False,
        'do_pos_pos': True,
        'var_method': 'jackknife',
        'use_randoms': True,
        'low_mem': False,
        }


    def read_nbin(self):
        # Get the number of halo bins in each axis
        with self.open_input("cluster_mag_halo_tomography") as f:
            meta = f['tomography'].attrs
            nm = meta['nm']
            nz = meta['nz']

        # Unlike regular TXPipe we have no source galaxies (i.e. galaxy shapes)
        source_list = []

        # but we do have two sets of density samples. The binned (in M and z) halos:
        lens_list = [f"{i}_{j}" for i in range(nz) for j in range(nm)]
        # and a single background sample:
        lens_list.append("bg")

        return source_list, lens_list


    def select_calculations(self, source_list, lens_list):
        calcs = []

        # In regular TXPipe this function selects all the three
        # 3x2pt measurements, but here all our measurements are
        # position-position.
        k = POS_POS
        for b1 in lens_list[:]:
            for b2 in lens_list:
                # We don't want to do a pair twice
                # The lenses in this list are actually
                # strings, but we can still compare them like this.
                if b1 <= b2:
                    calcs.append(b1, b2, k)

    def read_metadata(self):
        return {}


    def get_lens_catalog(self, bins):
        # We now have two different lens catalogs to choose from/
        if bins == "bg":
            cat = treecorr.Catalog(
                self.get_input("cluster_mag_background"),
                ext = f"/sample",
                ra_col = "ra",
                dec_col = "dec",
                ra_units='degree',
                dec_units='degree',
                patch_centers=self.get_input('patch_centers'),
            )
        else:
            cat = treecorr.Catalog(
                self.get_input("cluster_mag_halo_tomography"),
                ext = f"/lens/bin_{bins}",
                ra_col = "ra",
                dec_col = "dec",
                w_col = "weight",
                ra_units='degree',
                dec_units='degree',
                patch_centers=self.get_input('patch_centers'),
            )

        return cat


    def get_random_catalog(self, bins):
        # We get our randoms from cluster_mag_randoms, a single
        # unbinned catalog
        cat = treecorr.Catalog(
            self.get_input("random_cats"),
            ext = f"/randoms",
            ra_col = "ra",
            dec_col = "dec",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers'),
        )
        return cat

    def write_output(self, source_list, lens_list, meta, results):
        breakpoint()


