from .maps import TXBaseMaps
import numpy as np
from .base_stage import PipelineStage
from .mapping import Mapper, FlagMapper, BrightObjectMapper, DepthMapperDR1
from .data_types import MapsFile, HDFFile, ShearCatalog
from .utils import choose_pixelization
from .utils.calibration_tools import read_shear_catalog_type


class TXAuxiliaryMaps(TXBaseMaps):
    name = "TXAuxiliaryMaps"
    """
    This class generates:
        - depth maps
        - psf maps
        - bright object maps
        - flag maps
    """
    inputs = [
            ('photometry_catalog', HDFFile), # for mags etc
            ('shear_catalog', ShearCatalog), # for psfs
            ('shear_tomography_catalog', HDFFile), # for per-bin psf maps
            ('source_maps', MapsFile), # we copy the pixel scheme from here
    ]
    outputs = [
        ('aux_maps', MapsFile),
    ]

    config_options = {
        'chunk_rows': 100_000,
        'sparse': True,
        'psf_prefix': '',
        'flag_exponent_max': 8,
        'dilate': True,
        'psf_prefix': 'psf_',
        'bright_obj_threshold': 22.0, # The magnitude threshold for a object to be counted as bright
        'depth_band' : 'i',
        'snr_threshold': 10.0,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        'snr_delta':1.0,  # The range threshold +/- delta is used for finding objects at the boundary
    }
    # instead of reading from config we match the basic maps
    def choose_pixel_scheme(self):
        with self.open_input('source_maps', wrapper=True) as maps_file:
            pix_info = dict(maps_file.file['maps'].attrs)
        
        return choose_pixelization(**pix_info)



    def prepare_mappers(self, pixel_scheme):
        with self.open_input('shear_tomography_catalog') as f:
            nbin_source = f['tomography'].attrs['nbin_source']
        self.config['nbin_source'] = nbin_source # so it gets saved later
        source_bins = list(range(nbin_source))

        # For making psf_g1, psf_g2 maps, per source-bin
        psf_mapper = Mapper(pixel_scheme,
                            [],
                            source_bins,
                            do_lens=False,
                            sparse=self.config['sparse'])

        # for estimating depth per lens-bin
        depth_mapper = DepthMapperDR1(pixel_scheme,
                                      self.config['snr_threshold'],
                                      self.config['snr_delta'],
                                      sparse = self.config['sparse'],
                                      comm = self.comm)

        # for mapping bright star fractions, for masks
        brobj_mapper = BrightObjectMapper(pixel_scheme,
                                          self.config['bright_obj_threshold'],
                                          sparse = self.config['sparse'],
                                          comm = self.comm)

        # for mapping the density of flagged objects
        flag_mapper = FlagMapper(pixel_scheme, 
                                 self.config['flag_exponent_max'],
                                 sparse=self.config['sparse'])

        return psf_mapper, depth_mapper, brobj_mapper, flag_mapper


    def data_iterator(self):
        band = self.config['depth_band']
        psf_prefix = self.config['psf_prefix']

        shear_catalog_type = read_shear_catalog_type(self)

        if shear_catalog_type == 'metacal':
            shear_cols = [f'{psf_prefix}g1', f'{psf_prefix}g2', 'mcal_flags', 'weight']
        else:
            shear_cols = [f'{psf_prefix}g1', f'{psf_prefix}g2', 'flags', 'weight']


        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'photometry_catalog', # tag of input file to iterate through
            'photometry', # data group within file to look at
            ['ra', 'dec', 'extendedness', f'snr_{band}', f'mag_{band}'],
            # next file
            'shear_catalog', # tag of input file to iterate through
            'shear', # data group within file to look at
            shear_cols,
            # same file,different section
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
            # 'lens_tomography_catalog', # tag of input file to iterate through
            # 'tomography', # data group within file to look at
            # ['lens_bin'], # column(s) to read
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        psf_mapper, depth_mapper, brobj_mapper, flag_mapper = mappers

        band = self.config['depth_band']
        psf_prefix = self.config['psf_prefix']

        brobj_data = {
            'mag': data[f'mag_{band}'],
            'extendedness': data['extendedness'],
            'ra': data['ra'],
            'dec': data['dec'],
        }

        depth_data = {
            'mag': data[f'mag_{band}'],
            'snr': data[f'snr_{band}'],
            'ra': data['ra'],
            'dec': data['dec'],
        }

        psf_data = {
            'g1': data[f'{psf_prefix}g1'],
            'g2': data[f'{psf_prefix}g2'],
            'ra': data['ra'],
            'dec': data['dec'],
            'source_bin': data['source_bin'],
            'weight' : data['weight'],
        }

        flag_data = {
            'ra': data['ra'],
            'dec': data['dec'],
        }

        if self.config['shear_catalog_type'] == 'metacal':
            flag_data['flags'] = data['mcal_flags']
        else:
            flag_data['flags'] = data['flags']


        psf_mapper.add_data(psf_data)
        depth_mapper.add_data(depth_data)
        brobj_mapper.add_data(brobj_data)
        flag_mapper.add_data(flag_data)


    def finalize_mappers(self, pixel_scheme, mappers):
        psf_mapper, depth_mapper, brobj_mapper, flag_mapper = mappers

        # Four different mappers
        pix, _, g1, g2, var_g1, var_g2, weight = psf_mapper.finalize(self.comm)
        depth_pix, depth_count, depth, depth_var = depth_mapper.finalize(self.comm)
        brobj_pix, brobj_count, brobj_mag, brobj_mag_var = brobj_mapper.finalize(self.comm)
        flag_pixs, flag_maps = flag_mapper.finalize(self.comm)

        # Collect all the maps
        maps = {}

        if self.rank != 0:
            return maps


        # TODO: Could add density maps here, but need a clustering weight
        # mask to get an appropriate mean
        for b in psf_mapper.source_bins:
            maps['aux_maps', f'psf/g1_{b}'] = (pix, g1[b])
            maps['aux_maps', f'psf/g2_{b}'] = (pix, g1[b])
            maps['aux_maps', f'psf/var_g2_{b}'] = (pix, var_g1[b])
            maps['aux_maps', f'psf/var_g2_{b}'] = (pix, var_g1[b])
            maps['aux_maps', f'psf/lensing_weight_{b}'] = (pix, weight[b])


        maps['aux_maps', 'depth/depth'] = (depth_pix, depth)
        maps['aux_maps', 'depth/depth_count'] = (depth_pix, depth_count)
        maps['aux_maps', 'depth/depth_var'] = (depth_pix, depth_var)

        maps['aux_maps', "bright_objects/count"] =  (brobj_pix, brobj_count)

        for i, (p, m) in enumerate(zip(flag_pixs, flag_maps)):
            f = 2**i
            maps['aux_maps', f'flags/flag_{f}'] = (p, m)
            # also print out some stats
            t = m.sum()
            print(f"Map shows total {t} objects with flag {f}")

        return maps

