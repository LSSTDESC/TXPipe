from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile
from .utils.metacal import metacal_band_variants, metacal_variants
import numpy as np
import glob
import re

class TXMetacalGCRInput(PipelineStage):
    """
    This stage simulates metacal data and metacalibrated
    photometry measurements, starting from a cosmology catalogs
    of the kind used as an input to DC2 image and obs-catalog simulations.

    This is mainly useful for testing infrastructure in advance
    of the DC2 catalogs being available, but might also be handy
    for starting from a purer simulation.
    """
    name='TXMetacalGCRInput'

    inputs = []

    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
        ('star_catalog', HDFFile),
    ]

    config_options = {
        'cat_name': str,
    }

    def run(self):
        import GCRCatalogs
        import GCR
        import h5py
        # Open input data.  We do not treat this as a formal "input"
        # since it's the starting point of the whol pipeline and so is
        # not in a TXPipe format.
        cat_name = self.config['cat_name']
        cat = GCRCatalogs.load_catalog(cat_name)
        cat.master.use_cache = False

        # Total size is needed to set up the output file,
        # although in larger files it is a little slow to compute this.
        n = len(cat)
        print(f"Total catalog size = {n}")  

        available = cat.list_all_quantities()
        bands = []
        for b in 'ugrizy':
            if f'mcal_mag_{b}' in available:
                bands.append(b)

        # Columns that we will need.
        shear_cols = (['id', 'ra', 'dec', 'mcal_psf_g1', 'mcal_psf_g2', 'mcal_psf_T_mean', 'mcal_flags']
            + metacal_variants('mcal_g1', 'mcal_g2', 'mcal_T', 'mcal_s2n')
            + metacal_band_variants(bands, 'mcal_mag', 'mcal_mag_err')
        )

        # Input columns for photometry
        photo_cols = ['id', 'ra', 'dec']

        # Photometry columns (non-metacal)
        for band in 'ugrizy':
            photo_cols.append(f'{band}_mag')
            photo_cols.append(f'{band}_mag_err')
            photo_cols.append(f'snr_{band}_cModel')

        # Columns we need to load in for the star data - 
        # the measured object moments and the identifier telling us
        # if it was used in PSF measurement
        star_cols = [
            'id',
            'ra',
            'dec',
            'calib_psf_used',
            'Ixx',
            'Ixy',
            'Iyy',
            'IxxPSF',
            'IxyPSF',
            'IyyPSF',
        ]

        # For shear we just copy the input direct to the output
        shear_out_cols = shear_cols

        # For the photometry output we strip off the _cModeel suffix.
        photo_out_cols = [col[:-7] if col.endswith('_cModel') else col
                            for col in photo_cols]

        # The star output names are mostly different tot he input names
        star_out_cols = ['id', 'ra', 'dec', 
            'measured_e1', 'measured_e2',
            'model_e1', 'model_e2',
            'measured_T', 'model_T',
            'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']

        # eliminate duplicates before loading
        cols = list(set(shear_cols + photo_cols + star_cols))

        start = 0
        star_start = 0
        shear_output = None
        photo_output = None

        print("Skipping bad tract 2897 - remove this later!")

        # Loop through the data, as chunke natively by GCRCatalogs
        for data in cat.get_quantities(cols, return_iterator=True, native_filters='tract != 2897'):
            # Some columns have different names in input than output
            self.rename_columns(data)
            # The star ellipticities are derived from the measured moments for now
            star_data = self.compute_star_data(data)

            # First chunk of data we use to set up the output
            # It is easier this way (no need to check types etc)
            # if we change the column list
            if shear_output is None:
                shear_output = self.setup_output('shear_catalog', 'metacal', data, shear_out_cols, n)
                photo_output = self.setup_output('photometry_catalog', 'photometry', data, photo_out_cols, n)
                star_output  = self.setup_output('star_catalog', 'stars', star_data, star_out_cols, n)



            # Write out this chunk of data to HDF
            end = start + len(data['ra'])
            star_end = star_start + len(star_data['ra'])
            print(f"    Saving {start} - {end}")
            self.write_output(shear_output, 'metacal', shear_out_cols, start, end, data)
            self.write_output(photo_output, 'photometry', photo_out_cols, start, end, data)
            self.write_output(star_output,  'stars', star_out_cols,  star_start, star_end, star_data)
            start = end
            star_start = star_end

        # All done!
        photo_output.close()
        shear_output.close()
        star_output.close()

    def rename_columns(self, data):
        for band in 'ugrizy':
            data[f'snr_{band}'] = data[f'snr_{band}_cModel']
            del data[f'snr_{band}_cModel']

    def setup_output(self, name, group, cat, cols, n):
        import h5py
        f = self.open_output(name)
        g = f.create_group(group)
        for name in cols:
            g.create_dataset(name, shape=(n,), dtype=cat[name].dtype)
        return f


    def write_output(self, output_file, group_name, cols, start, end, data):
        g = output_file[group_name]
        for name in cols:
            g[name][start:end] = data[name]


    def compute_star_data(self, data):
        star_data = {}
        # We specifically use the stars chosen for PSF measurement
        star = data['calib_psf_used']

        # General columns
        star_data['ra'] = data['ra'][star]
        star_data['dec'] = data['dec'][star]
        star_data['id'] = data['id'][star]
        for band in 'ugrizy':
            star_data[f'{band}_mag'] = data[f'{band}_mag'][star]

        for b in 'ugrizy':
            star_data[f'{b}_mag'] = data[f'{b}_mag'][star]

        # HSM reports moments.  We convert these into
        # ellipticities.  We do this for both the star shape
        # itself and the PSF model.
        kinds = [
            ('', 'measured_'),
            ('PSF', 'model_')
        ]

        for in_name, out_name in kinds:
            # Pulling out the correct moment columns
            Ixx = data[f'Ixx{in_name}'][star]
            Iyy = data[f'Iyy{in_name}'][star]
            Ixy = data[f'Ixy{in_name}'][star]

            # Conversion of moments to e1, e2
            T = Ixx + Iyy
            e = (Ixx - Iyy + 2j * Ixy) / (Ixx + Iyy)
            e1 = e.real
            e2 = e.imag

            # save to output
            star_data[f'{out_name}e1'] = e1
            star_data[f'{out_name}e2'] = e2
            star_data[f'{out_name}T'] = T

        return star_data



class TXGCRTwoCatalogInput(TXMetacalGCRInput):
    """

    """
    name='TXGCRTwoCatalogInput'

    inputs = []

    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
        ('star_catalog', HDFFile),
    ]

    config_options = {
        'photo_dir': '/global/projecta/projectdirs/lsst/production/DC2_ImSim/Run2.1i/dpdd/calexp-v1:coadd-dr1b-v1/object_table_parquet',
        'photo_is_hdf5': False,
        'metacal_dir': '/global/projecta/projectdirs/lsst/production/DC2_ImSim/Run2.1i/dpdd/coadd-dr1b-v1:metacal-dr1b-v2/metacal_table_summary',
    }

    # dirs for in2p3
    # '/sps/lsst/data/desc/catalogs/Run2.1i/dpdd/calexp-v1:coadd-dr1b-v1/object_table_parquet',
    # "/sps/lsst/dataproducts/desc/DC2/Run2.1i/w_2019_19-v2/dpdd/coadd-dr1b-v1:metacal-dr1b-v2/metacal_table_summary",

    def run(self):
        import GCR
        import GCRCatalogs
        import h5py
        # Open input data.  We do not treat this as a formal "input"
        # since it's the starting point of the whole pipeline and so is
        # not in a TXPipe format.
        # shear_cat = GCRCatalogs.load_catalog('dc2_object_run2.1.1i_with_metacal.yaml')
        # photo_cat = shear_cat
        metacal_dir = self.config['metacal_dir']
        photo_dir = self.config['photo_dir']

        metacal_config = {'base_dir': metacal_dir}
        photo_config = {'base_dir': photo_dir}

        if self.config['photo_is_hdf5']:
            photo_config['filename_pattern'] = 'object_tract_\\d+\\.hdf5'
            photo_config['subclass_name'] = 'dc2_object.DC2ObjectCatalog'

        shear_cat = GCRCatalogs.load_catalog('dc2_metacal_griz_run2.1i_dr1b', metacal_config)
        photo_cat = GCRCatalogs.load_catalog('dc2_object_run2.1i_dr1b', photo_config)
        # ?? :
        #'filename_pattern': 'object_tract_\\d+\\.hdf5',
        #'subclass_name': 'dc2_object.DC2ObjectCatalog'
        composite_cat = GCR.CompositeCatalog(
            [shear_cat, photo_cat], 
            ['metacal', 'object'],
        )



        available = shear_cat.list_all_quantities()
        bands = []
        for b in 'ugrizy':
            if f'mcal_mag_{b}' in available:
                bands.append(b)

        # Columns that we will need.
        shear_cols = (['id', 'mcal_psf_g1', 'mcal_psf_g2', 'mcal_psf_T_mean', 'mcal_flags']
            + metacal_variants('mcal_g1', 'mcal_g2', 'mcal_T', 'mcal_s2n')
            + metacal_band_variants(bands, 'mcal_mag', 'mcal_mag_err')
        )

        # Input columns for photometry
        photo_cols = ['id', 'ra', 'dec']
        photo_out_cols = photo_cols[:]
        # Photometry columns (non-metacal)
        for band in 'ugrizy':
            photo_cols.append(f'mag_{band}')
            photo_cols.append(f'magerr_{band}')
            photo_cols.append(f'{band}_modelfit_CModel_instFluxErr')
            photo_cols.append(f'{band}_modelfit_CModel_instFlux')

            photo_out_cols.append(f'{band}_mag')
            photo_out_cols.append(f'{band}_mag_err')
            photo_out_cols.append(f'snr_{band}')

            
        # Columns we need to load in for the star data - 
        # the measured object moments and the identifier telling us
        # if it was used in PSF measurement
        star_cols = [
            'id',
            'ra',
            'dec',
            'calib_psf_used',
            'Ixx',
            'Ixy',
            'Iyy',
            'IxxPSF',
            'IxyPSF',
            'IyyPSF',
        ]

        # For shear we just copy the input direct to the output
        shear_out_cols = shear_cols + ['ra', 'dec']

        # The star output names are mostly different tot he input names
        star_out_cols = ['id', 'ra', 'dec', 
            'measured_e1', 'measured_e2',
            'model_e1', 'model_e2',
            'measured_T', 'model_T',
            'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag',
        ]

        print("Counting stars")
        is_star = photo_cat.get_quantities(['calib_psf_used'])['calib_psf_used']
        nstar = is_star.sum()
        print(f"Found {nstar} stars")

        photo_cols = list(set(photo_cols + star_cols))
        cols = photo_cols + shear_cols
        # eliminate duplicates before loading
        start = 0
        star_start = 0
        shear_output = None
        photo_output = None

        print("Counting all objects")
        n = len(composite_cat)

        # Loop through the data, as chunke natively by GCRCatalogs
        print("Starting data load")
        for data in composite_cat.get_quantities(cols, return_iterator=True):
            # Some columns have different names in input than output
            chunk_size = len(data['id'])
            print(f"Loaded chunk of size {chunk_size}")
            self.rename_columns(data)
            # The star ellipticities are derived from the measured moments for now
            star_data = self.compute_star_data(data)

            # First chunk of data we use to set up the output
            # It is easier this way (no need to check types etc)
            # if we change the column list
            if shear_output is None:
                shear_output = self.setup_output('shear_catalog', 'metacal', data, shear_out_cols, n)
                photo_output = self.setup_output('photometry_catalog', 'photometry', data, photo_out_cols, n)
                star_output  = self.setup_output('star_catalog', 'stars', star_data, star_out_cols, nstar)

            # Write out this chunk of data to HDF
            end = start + len(data['ra'])
            star_end = star_start + len(star_data['ra'])
            print(f"    Saving {start} - {end}")
            self.write_output(shear_output, 'metacal', shear_out_cols, start, end, data)
            self.write_output(photo_output, 'photometry', photo_out_cols, start, end, data)
            self.write_output(star_output,  'stars', star_out_cols,  star_start, star_end, star_data)
            start = end
            star_start = star_end

        # All done!
        photo_output.close()
        shear_output.close()
        star_output.close()

    def rename_columns(self, data):
        for band in 'ugrizy':
            data[f'{band}_mag'] = data[f'mag_{band}']
            del data[f'mag_{band}']
            data[f'{band}_mag_err'] = data[f'magerr_{band}']
            del data[f'magerr_{band}']
            data[f'snr_{band}'] = data[f'{band}_modelfit_CModel_instFlux'] / data[f'{band}_modelfit_CModel_instFluxErr']
            del data[f'{band}_modelfit_CModel_instFlux']
            del data[f'{band}_modelfit_CModel_instFluxErr']

        

# response to an old Stack Overflow question of mine:
# https://stackoverflow.com/questions/33529057/indices-that-intersect-and-sort-two-numpy-arrays
def intersecting_indices(x, y):
    u_x, u_idx_x = np.unique(x, return_index=True)
    u_y, u_idx_y = np.unique(y, return_index=True)
    i_xy = np.intersect1d(u_x, u_y, assume_unique=True)
    i_idx_x = u_idx_x[np.in1d(u_x, i_xy, assume_unique=True)]
    i_idx_y = u_idx_y[np.in1d(u_y, i_xy, assume_unique=True)]
    return i_idx_x, i_idx_y

