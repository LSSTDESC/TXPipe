from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, MapsFile, HDFFile
import numpy as np
from .utils import unique_list, choose_pixelization
from .mapping import Mapper, FlagMapper

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

class TXBaseMaps(PipelineStage):
    """
    This is an abstract base mapping class, which other subclasses
    inherit from to use the same basic structure, which is:
    - select pixelization
    - prepare some mapper objects
    - iterate through selected columns
        - update each mapper with each chunk
    - finalize the mappers
    - save the maps
    """
    name = "TXBaseMaps"
    inputs = []
    outputs = []
    config_options = {
        'chunk_rows': 100000,  # The number of rows to read in each chunk of data at a time
        'pixelization': 'healpix', # The pixelization scheme to use, currently just healpix
        'nside': 0,   # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
        'sparse': True,   # Whether to generate sparse maps - faster and less memory for small sky areas,
        'ra_cent': np.nan,  # These parameters are only required if pixelization==tan
        'dec_cent': np.nan,
        'npix_x':-1,
        'npix_y':-1,
        'pixel_size': np.nan, # Pixel size of pixelization scheme
        'true_shear' : False,
    }

    def run(self):
        # Read input configuration information
        # and select pixel scheme. Also save the scheme
        # metadata so we can save it later
        pixel_scheme = self.choose_pixel_scheme()
        self.config.update(pixel_scheme.metadata)

        # Initialize our maps
        mappers = self.prepare_mappers(pixel_scheme)

        # Loop through the data
        for s, e, data in self.data_iterator():
            # Give an idea of progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Build up any maps we are using
            self.accumulate_maps(pixel_scheme, data, mappers)

        # Finalize and output the maps
        maps = self.finalize_mappers(pixel_scheme, mappers)
        if self.rank == 0:
            self.save_maps(pixel_scheme, maps)

    def choose_pixel_scheme(self):
        """
        Subclasses can override to instead load pixelization
        from an existing map
        """
        return choose_pixelization(**self.config)

    def prepare_mappers(self, pixel_scheme):
        """
        Subclasses must override to init any mapper objects
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def data_iterator(self):
        """
        Subclasses must override to create an iterator looping over
        input data
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def accumulate_maps(self, pixel_scheme, data, mappers):
        """
        Subclasses must override to supply the next chunk "data" to
        their mappers
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def finalize_mappers(self, pixel_scheme, mappers):
        """
        Subclasses must override to finalize their maps and return
        a dictionary of (output_tag, map_name) -> (pixels, values)
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")


    def save_maps(self, pixel_scheme, maps):
        """
        Subclasses can use this directly, by generating maps as described
        in finalize_mappers
        """
        # Find and open all output files
        tags = unique_list([outfile for outfile, _ in maps.keys()])
        output_files = {tag: self.open_output(tag, wrapper=True) for tag in tags if tag.endswith('maps')}

        # add a maps section to each
        for output_file in output_files.values():
            output_file.file.create_group("maps")
            output_file.file['maps'].attrs.update(self.config)

        # same the relevant maps in each
        for (tag, map_name), (pix, map_data) in maps.items():
            output_files[tag].write_map(map_name, pix, map_data, self.config)


class TXSourceMaps(TXBaseMaps):
    name = "TXSourceMaps"
    inputs = [
        ('shear_catalog', HDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('source_maps', MapsFile),
    ]

    def prepare_mappers(self, pixel_scheme):
        # open and return a single mapper
        # read nbin_source
        with self.open_input('shear_tomography_catalog') as f:
            nbin_source = f['tomography'].attrs['nbin_source']
        # store in config so it is saved later
        self.config['nbin_source'] = nbin_source

        # create basic mapper object
        source_bins = list(range(nbin_source))
        lens_bins = []
        mapper = Mapper(
                    pixel_scheme,
                    lens_bins,
                    source_bins,
                    do_lens=False,
                    sparse=self.config['sparse']
                )
        return [mapper]

    def data_iterator(self):
        # read shear cols and 

        # can optionally read truth values
        if self.config['true_shear']:
            shear_cols = ['true_g1', 'true_g1', 'ra', 'dec', 'weight']
        else:
            shear_cols = ['mcal_g1', 'mcal_g2', 'ra', 'dec', 'weight']

        # use utility function that combines data chunks
        # from different files
        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'shear_catalog', # tag of input file to iterate through
            'metacal', # data group within file to look at
            shear_cols, # column(s) to read
            # next file
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        # rename columns
        if self.config['true_shear']:
            data['g1'] = data['true_g1']
            data['g2'] = data['true_g2']
        else:
            data['g1'] = data['mcal_g1']
            data['g2'] = data['mcal_g2']
        # send data to map
        mapper = mappers[0]
        mapper.add_data(data)
    
    def finalize_mappers(self, pixel_scheme, mappers):
        # only one mapper here - we call its finalize method
        # to collect everything
        mapper = mappers[0]
        pix, _, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps['source_maps', f'g1_{b}'] = (pix, g1[b])
            maps['source_maps', f'g2_{b}'] = (pix, g2[b])
            maps['source_maps', f'var_g1_{b}'] = (pix, var_g1[b])
            maps['source_maps', f'var_g1_{b}'] = (pix, var_g1[b])
            maps['source_maps', f'lensing_weight_{b}'] = (pix, var_g1[b])
    
        return maps

class TXLensMaps(TXBaseMaps):
    name = "TXLensMaps"
    inputs = [
        ('photometry_catalog', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('lens_maps', MapsFile),
    ]
    def prepare_mappers(self, pixel_scheme):
        # read nbin_lens and save
        with self.open_input('lens_tomography_catalog') as f:
            nbin_lens = f['tomography'].attrs['nbin_lens']
        self.config['nbin_lens'] = nbin_lens

        # create lone mapper
        lens_bins = list(range(nbin_lens))
        source_bins = []
        mapper = Mapper(
                    pixel_scheme,
                    lens_bins,
                    source_bins,
                    do_g=False,
                    sparse=self.config['sparse']
                )
        return [mapper]

    def data_iterator(self):
        print("TODO: no lens weights here")
        # iterate through tomography and photometry
        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'photometry_catalog', # tag of input file to iterate through
            'photometry', # data group within file to look at
            ['ra', 'dec'], # column(s) to read
            # next file
            'lens_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['lens_bin'], # column(s) to read
            # another section in the same file
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        # no need to rename cols here since just ra, dec, lens_bin
        mapper = mappers[0]
        mapper.add_data(data)
    
    def finalize_mappers(self, pixel_scheme, mappers):
        # Again just the one mapper
        mapper = mappers[0]
        pix, ngal, _, _, _, _, _ = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps['lens_maps', f'ngal_{b}'] = (pix, ngal[b])
    
        return maps


class TXExternalLensMaps(TXLensMaps):
    """
    Same as TXLensMaps except it reads from an external
    lens catalog.
    """
    name = "TXExternalLensMaps"

    inputs = [
        ('lens_catalog', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
    ]

    def data_iterator(self):
        print("TODO: no lens weights here")
        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'lens_catalog', # tag of input file to iterate through
            'photometry', # data group within file to look at
            ['ra', 'dec'], # column(s) to read
            # next file
            'lens_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['lens_bin'], # column(s) to read
            # another section in the same file
        )




class TXMainMaps(TXSourceMaps, TXLensMaps):
    """
    Combined source and photometric lens maps, from the
    same photometry catalog
    """
    name = "TXMainMaps"

    inputs = [
        ('photometry_catalog', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_catalog', HDFFile),
]
    outputs = [
        ('lens_maps', MapsFile),
        ('source_maps', MapsFile),
    ]

    def data_iterator(self):
        # This is just the combination of
        # the source and lens map columns
        print("TODO: no lens weights here")

        # 
        if self.config['true_shear']:
            shear_cols = ['true_g1', 'true_g2', 'ra', 'dec', 'weight']
        else:
            shear_cols = ['mcal_g1', 'mcal_g2', 'ra', 'dec', 'weight']

        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'photometry_catalog', # tag of input file to iterate through
            'photometry', # data group within file to look at
            ['ra', 'dec'], # column(s) to read
            # next file
            'shear_catalog', # tag of input file to iterate through
            'metacal', # data group within file to look at
            shear_cols, # column(s) to read
            # next file
            'lens_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['lens_bin'], # column(s) to read
            # same file,different section
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
        )

    def prepare_mappers(self, pixel_scheme):
        # read both nbin values
        with self.open_input('shear_tomography_catalog') as f:
            nbin_source = f['tomography'].attrs['nbin_source']

        with self.open_input('lens_tomography_catalog') as f:
            nbin_lens = f['tomography'].attrs['nbin_lens']

        self.config['nbin_source'] = nbin_source
        self.config['nbin_lens'] = nbin_lens

        source_bins = list(range(nbin_source))
        lens_bins = list(range(nbin_lens))

        # still a single mapper doing source and lens
        mapper = Mapper(
                    pixel_scheme,
                    lens_bins,
                    source_bins,
                    sparse=self.config['sparse'])
        return [mapper]

    # accumulate_maps is inherited from TXSourceMaps because
    # that appears first in the list above

    def finalize_mappers(self, pixel_scheme, mappers):
        mapper = mappers[0]
        pix, ngal, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        # Now both loops, source and lens
        for b in mapper.lens_bins:
            maps['lens_maps', f'ngal_{b}'] = (pix, ngal[b])

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps['source_maps', f'g1_{b}'] = (pix, g1[b])
            maps['source_maps', f'g2_{b}'] = (pix, g2[b])
            maps['source_maps', f'var_g1_{b}'] = (pix, var_g1[b])
            maps['source_maps', f'var_g1_{b}'] = (pix, var_g1[b])
            maps['source_maps', f'lensing_weight_{b}'] = (pix, var_g1[b])

        return maps




class TXDensityMaps(PipelineStage):
    """
    Convert n_gal maps to overdensity delta maps
    delta = (ngal - <ngal>) / <ngal>

    This has to be separate from the lens mappers above
    because it requires the mask, which is created elsewhere
    (right now in masks.py)
    """
    name = "TXDensityMaps"
    inputs = [
        ('lens_maps', MapsFile),
        ('mask', MapsFile),
    ]
    outputs = [
        ('density_maps', MapsFile),
    ]

    def run(self):
        import healpy
        # Read the mask
        with self.open_input('mask', wrapper=True) as f:
            mask = f.read_map('mask')

        # set unseen pixels to weight zero
        mask[mask == healpy.UNSEEN] = 0
        pix = np.where(mask > 0)[0]

        # Read the count maps
        with self.open_input('lens_maps', wrapper=True) as f:
            meta = dict(f.file['maps'].attrs)
            nbin_lens = meta['nbin_lens']
            ngal_maps = [f.read_map(f'ngal_{b}') for b in range(nbin_lens)]

        # Convert count maps into density maps
        density_maps = []
        for i, ng in enumerate(ngal_maps):
            # Convert the number count maps to overdensity maps.
            # First compute the overall mean object count per bin.
            # mean clustering galaxies per pixel in this map
            mu = np.average(ng, weights=mask)
            # and then use that to convert to overdensity
            d = (ng - mu) / mu
            # remove nans
            d[mask==0] = 0
            density_maps.append(d)

        # write output
        with self.open_output('density_maps', wrapper=True) as f:
            # create group and save metadata there too.
            f.file.create_group("maps")
            f.file['maps'].attrs.update(meta)
            # save each density map
            for i, rho in enumerate(density_maps):
                f.write_map(f'delta_{i}', pix, rho[pix], meta)
    



