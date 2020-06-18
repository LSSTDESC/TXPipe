from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, MapsFile, HDFFile
import numpy as np
from .utils import unique_list, choose_pixelization
from .mapping import Mapper, FlagMapper

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

class TXBaseMaps(PipelineStage):
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
        return choose_pixelization(**self.config)


    def prepare_mappers(self):
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def save_maps(self, pixel_scheme, maps):
        tags = unique_list([outfile for outfile, _ in maps.keys()])
        output_files = {tag: self.open_output(tag, wrapper=True) for tag in tags if tag.endswith('maps')}

        for output_file in output_files.values():
            output_file.file.create_group("maps")
            output_file.file['maps'].attrs.update(self.config)

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
        with self.open_input('shear_tomography_catalog') as f:
            nbin_source = f['tomography'].attrs['nbin_source']
        self.config['nbin_source'] = nbin_source

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
        if self.config['true_shear']:
            shear_cols = ['true_g', 'ra', 'dec', 'weight']
        else:
            shear_cols = ['mcal_g1', 'mcal_g2', 'ra', 'dec', 'weight']

        return self.combined_iterators(
            self.config['chunk_rows'],
            # first file
            'shear_catalog', # tag of input file to iterate through
            'metacal', # data group within file to look at
            shear_cols, # column(s) to read
            # next gile
            'shear_tomography_catalog', # tag of input file to iterate through
            'metacal_response', # data group within file to look at
            ['R_gamma'], # column(s) to read
            # 
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        data['g1'] = data['mcal_g1']
        data['g2'] = data['mcal_g2']
        mapper = mappers[0]
        mapper.add_data(data) # update this - out of date
    
    def finalize_mappers(self, pixel_scheme, mappers):
        mapper = mappers[0]
        pix, _, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)
        maps = {}

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
        with self.open_input('lens_tomography_catalog') as f:
            nbin_lens = f['tomography'].attrs['nbin_lens']
        self.config['nbin_lens'] = nbin_lens

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
        mapper = mappers[0]
        mapper.add_data(data) # update this - out of date
    
    def finalize_mappers(self, pixel_scheme, mappers):
        mapper = mappers[0]
        pix, ngal, _, _, _, _, _ = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        # TODO: Could add density maps here, but need a clustering weight
        # mask to get an appropriate mean

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps['lens_maps', f'ngal_{b}'] = (pix, ngal[b])
    
        return maps


class TXExternalLensMaps(TXLensMaps):
    name = "TXExternalLensMaps"
    inputs = [
        ('lens_catalog', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('lens_maps', MapsFile),
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
    def finalize_mappers(self, pixel_scheme, mappers):
        mapper = mappers[0]
        pix, ngal, _, _, _, _, _ = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        # TODO: Could add density maps here, but need a clustering weight
        # mask to get an appropriate mean

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps['lens_maps', f'ngal_{b}'] = (pix, ngal[b])
    
        return maps





class TXMainMaps(TXSourceMaps, TXLensMaps):
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
        print("TODO: no lens weights here")

        if self.config['true_shear']:
            shear_cols = ['true_g', 'ra', 'dec', 'weight']
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
        with self.open_input('shear_tomography_catalog') as f:
            nbin_source = f['tomography'].attrs['nbin_source']

        with self.open_input('lens_tomography_catalog') as f:
            nbin_lens = f['tomography'].attrs['nbin_lens']

        self.config['nbin_source'] = nbin_source
        self.config['nbin_lens'] = nbin_lens

        source_bins = list(range(nbin_source))
        lens_bins = list(range(nbin_lens))

        mapper = Mapper(
                    pixel_scheme,
                    lens_bins,
                    source_bins,
                    sparse=self.config['sparse'])
        return [mapper]


    def finalize_mappers(self, pixel_scheme, mappers):
        mapper = mappers[0]
        pix, ngal, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        # TODO: Could add density maps here, but need a clustering weight
        # mask to get an appropriate mean
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
            # Maybe we should do this in the mapping code itself?
            # mean clustering galaxies per pixel in this map
            mu = np.average(ng, weights=mask)
            # and then use that to convert to overdensity
            d = (ng - mu) / mu
            # remove nans
            d[mask==0] = 0
            density_maps.append(d)

        with self.open_output('density_maps', wrapper=True) as f:
            f.file.create_group("maps")
            f.file['maps'].attrs.update(meta)
            for i, rho in enumerate(density_maps):
                f.write_map(f'delta_{i}', pix, rho[pix], meta)
    



