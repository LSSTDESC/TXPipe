import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile, TextFile, SACCFile
import re
import time
import traceback
from ...twopoint import TXTwoPoint, POS_POS


class CMCorrelations(TXTwoPoint):
    name = "CMCorrelations"

    inputs = [
         ("cluster_mag_halo_tomography", HDFFile),
         ("cluster_mag_background", HDFFile),
         ("patch_centers", TextFile),
         ("random_cats", HDFFile),
    ]
    outputs = [
        ('cluster_mag_correlations', SACCFile),
    ]

    config_options = {
        'calcs': [POS_POS],
        'min_sep': 0.5,
        'max_sep': 300.,
        'nbins': 9,
        'bin_slop': 0.0,
        'sep_units': 'arcmin',
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
        'do_halo_cross': False,
        'patch_dir': './cache/patches',
    }


    def read_nbin(self):
        # Get the number of halo bins in each axis
        with self.open_input("cluster_mag_halo_tomography") as f:
            meta = f['lens'].attrs
            nm = meta['nm']
            nz = meta['nz']

        # Unlike regular TXPipe we have no source galaxies (i.e. galaxy shapes)
        source_list = []

        # but we do have two sets of density samples. The binned (in M and z) halos:
        lens_list = [f"{i}_{j}" for i in range(nz) for j in range(nm)]
        # and a single background sample:
        lens_list.insert(0, "bg")

        return source_list, lens_list


    def select_calculations(self, source_list, lens_list):
        calcs = []

        # In regular TXPipe this function selects all the three
        # 3x2pt measurements, but here all our measurements are
        # position-position.
        k = POS_POS

        if self.config['do_halo_cross']:
            # All the cross-correlations - takes ages
            for b1 in lens_list[:]:
                for b2 in lens_list:
                    # We don't want to do a pair twice
                    # The lenses in this list are actually
                    # strings, but we can still compare them like this.
                    if b1 <= b2:
                        calcs.append([b1, b2, k])
        else:
            # If we are not doing correlations between halo bins
            # then we just want the auto-correlation for the non-bg bins.
            # We want bg x everything, including bg x bg
            for b1 in lens_list[:]:
                if b1 == "bg":
                    for b2 in lens_list:
                        calcs.append([b1, b2, k])
                else:
                    calcs.append([b1, b1, k])
                        

        return calcs

    def read_metadata(self):
        meta = {}
        # Read per-bin inforation
        with self.open_input("cluster_mag_halo_tomography") as f:
            g = f['lens']
            for key in g.keys():
                meta[key] = dict(g[key].attrs)

        # And also general information 
        with self.open_input("cluster_mag_halo_tomography") as f:
            meta['nm'] = f['lens'].attrs['nm']
            meta['nz'] = f['lens'].attrs['nz']

        return meta


    def get_lens_catalog(self, bins):
        import treecorr
        self.memory_report(f"get lens {bins}")
        # We now have two different lens catalogs to choose from
        if bins == "bg":
            cat = treecorr.Catalog(
                self.get_input("cluster_mag_background"),
                ext = f"/sample",
                ra_col = "ra",
                dec_col = "dec",
                ra_units='degree',
                dec_units='degree',
                patch_centers=self.get_input('patch_centers'),
                save_patch_dir=self.get_patch_dir('cluster_mag_background', bins),
            )
        else:
            cat = treecorr.Catalog(
                self.get_input("cluster_mag_halo_tomography"),
                ext = f"/lens/bin_{bins}",
                ra_col = "ra",
                dec_col = "dec",
                ra_units='degree',
                dec_units='degree',
                patch_centers=self.get_input('patch_centers'),
                save_patch_dir=self.get_patch_dir('cluster_mag_halo_tomography', bins),
            )

        return cat


    def get_random_catalog(self, bins):
        import treecorr
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
            save_patch_dir=self.get_patch_dir('random_cats', bins),
        )
        return cat

    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        import treecorr

        # Create a sacc for our output data and
        # save some metadata in it
        S = sacc.Sacc()
        S.metadata['nm'] = meta['nm']
        S.metadata['nz'] = meta['nz']

        # Record the names of all our bins. At some point we may
        # want to load the n(z) here and put it in the output too
        for bins in lens_list:
            bin_output_name = "background" if bins == "bg" else f'halo_{bins}'
            S.add_tracer('misc', bin_output_name)


        for d in results:
            # Name of the pair of bins used here. Probably the same
            # unless we have set do_halo_cross=True
            tracer1 = f'background' if d.i == "bg" else f'halo_{d.i}'
            tracer2 = f'background' if d.j == "bg" else f'halo_{d.j}'

            # select name of the data type, and any metadata to store
            if d.i == "bg" and d.j == "bg":
                corr_type = "galaxy_density_xi"
                tags = {}
            elif d.i == "bg":
                corr_type = "halo_galaxy_density_xi"
                tags = meta[f'bin_{d.j}']
            elif d.i == d.j:
                corr_type = "halo_halo_density_xi"
                tags = meta[f'bin_{d.i}']
            else:
                corr_type = "halo_halo_density_xi"
                # combined metadata for both
                tags = {
                    **{"{k}_bin2": "{v}_bin2" for k, v in meta[f'bin_{d.i}'].items()},
                    **{"{k}_bin2": "{v}_bin2" for k, v in meta[f'bin_{d.j}'].items()},
                }

            # Other numbers to save
            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight
            xi = d.object.xi
            err = np.sqrt(d.object.varxi)
            n = len(xi)

            # Add all our data points to the output file
            for i in range(n):
                S.add_data_point(corr_type, (tracer1, tracer2), xi[i],
                    theta=theta[i], error=err[i], weight=weight[i], **tags)         

        # Compute the covariance with treecorr and add it to the output
        cov = treecorr.estimate_multi_cov(
            [d.object for d in results],
            self.config['var_method']
        )
        S.add_covariance(cov)

        # Save results to FITS format
        S.save_fits(self.get_output("cluster_mag_correlations"))
