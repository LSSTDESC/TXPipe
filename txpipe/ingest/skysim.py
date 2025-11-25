from tracemalloc import start
from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog
from ceci.config import StageParameter
from .mock_tools import add_lsst_like_noise, make_metadetect_catalog
import numpy as np
import glob


class TableWriter:
    def __init__(self, group, initial_size=1_000_000):
        import h5py
        self.group = group
        self.created_datasets = False
        self.initial_size = initial_size
        self.keys = []
        self.current_size = {}
        self.start = 0

    def do_size_check(self, data):
        sz = None
        for key, value in data.items():
            if sz is None:
                sz = len(value)
            elif sz != len(value):
                raise ValueError("All arrays to write must have the same length")
        return sz
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.resize(self.start)

    def create_datasets(self, data):
        size = max(self.initial_size, len(data[next(iter(data))]))
        shape = (size,)
        for key, value in data.items():
            maxshape = (None,)
            self.group.create_dataset(key, shape, maxshape=maxshape, dtype=value.dtype)
        self.current_size = size
        self.created_datasets = True
        self.keys = list(data.keys())

    def resize(self, new_size):
        for key in self.keys:
            self.group[key].resize((new_size,))
        self.current_size = new_size

    def write(self, data):
        sz = self.do_size_check(data)
        
        if not self.created_datasets:
            self.create_datasets(data)

        end = self.start + sz

        if end > self.current_size:
            new_size = max(self.current_size * 2, end)
            self.resize(new_size)

        for key in self.keys:
            value = data[key]
            self.group[key][self.start:end] = value
        self.start = end


class ShearTableWriter:
    def __init__(self, filename, initial_size=1_000_000):
        import h5py
        self.file = h5py.File(filename, "w")
        self.group = self.file.create_group("shear")
        self.writers = {}
        for variant in ["00", "1p", "1m", "2p", "2m"]:
            group = self.file.create_group(f"shear/{variant}")
            self.writers[variant] = TableWriter(group, initial_size=initial_size)
    
    def __enter__(self):
        for writer in self.writers.values():
            writer.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for writer in self.writers.values():
            writer.__exit__(exc_type, exc_value, traceback)
        self.file.close()

    def write(self, data):
        for variant in self.writers:
            tag = variant + "/"
            data_variant = {key.removeprefix(tag): value for key, value in data.items() if key.startswith(tag)}
            self.writers[variant].write(data_variant)





class TXIngestRomanRubin(PipelineStage):
    """Ingest skysim simulation data for TXPipe processing.

    This stage reads in skysim simulation data files and converts them
    into the internal shear catalog format used by TXPipe. It handles
    the necessary metadata and ensures compatibility with downstream
    analysis stages.
    """
    name = "TXIngestRomanRubin"
    inputs = []
    parallel = False
    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", PhotometryCatalog),
    ]
    config_options = {
        "xgal_dir_name": StageParameter(str, default="/global/cfs/cdirs/lsst/shared/xgal/roman-rubin/roman_rubin_2023_v1.1.3", msg="Directory name for skysim xgal files"),
        "delta_gamma": StageParameter(float, default=0.02, msg="Delta gamma value for metadetect response calculations"),
        "year": StageParameter(int, default=1, msg="Number of years of LSST observations to simulate photometric noise for"),
        "response_type": StageParameter(str, default="unit", msg="Type of response to apply for metadetect"),
        "snr_cut": StageParameter(float, default=5.0, msg="SNR cut for overall detection"),
        "T_ratio_cut": StageParameter(float, default=0.5, msg="T/PSF_T cut for metadetect catalog"),
        "random_seed": StageParameter(int, default=0, msg="Random seed"),
    }


    def run(self):
        import h5py


        file_format = "roman_rubin_2023_*.hdf5"
        files = glob.glob(f"{self.config.xgal_dir_name}/{file_format}")

        # This is fixed for Roman-Rubin sims to the usual LSST bands.
        bands = "ugrizy"
        shear_bands = "griz"

        response_type = self.config["response_type"]
        year = self.config["year"]
        delta_gamma = self.config["delta_gamma"]

        shear_filename = self.get_output("shear_catalog")

        # TODO: Update this to a more reasonable size once we know how big
        # the catalogs will be. This is just a guess for now and things will be
        # resized as needed.
        size = 10_000_000

        photo_file = self.open_output("photometry_catalog")
        photo_group = photo_file.create_group("photometry")

        snr_cut = self.config["snr_cut"]
        T_ratio_cut = self.config["T_ratio_cut"]
        rng = np.random.default_rng(self.config['random_seed'])
        
        with ShearTableWriter(shear_filename, initial_size=size) as shear_writer, \
            TableWriter(photo_group, initial_size=size) as photo_writer:
            nfile = len(files)
            for j, filename in enumerate(files):
                print(f"Processing file {j+1}/{nfile}: {filename}")
                with h5py.File(filename, "r") as f:
                    nkey = len(f.keys()) - 1

                    # The keys here are mostly healpix pixel indices.
                    for i, key in enumerate(f.keys()):
                        if key == "metaData":
                            continue

                        group = f[key]

                        if "ra" not in group.keys():
                            print(f"    No data - skipping healpix pixel {i+1}/{nkey}: {key} ")
                            continue
                        print(f"    Processing healpix pixel {i+1}/{nkey}: {key}")
                        data = extract_roman_rubin_truth_info(group, bands)
                        add_lsst_like_noise(data, rng, year=year)
                        shear_data = make_metadetect_catalog(data, response_type, delta_gamma, shear_bands, rng, snr_cut=snr_cut, T_ratio_cut=T_ratio_cut)
                        photo_data = make_photo_cuts(data, snr_cut)
                        shear_writer.write(shear_data)
                        photo_writer.write(photo_data)
        
        photo_file.close()


def make_photo_cuts(data, bands, snr_cut):
    # check if object is detected in any band
    detected = np.zeros_like(data["id"], dtype=bool)
    for b in bands:
        snr = data[f"snr_{b}"]
        detected |= (snr >= snr_cut)

    output = {}
    for key, value in data.items():
        output[key] = value[detected]
    return output


def extract_roman_rubin_truth_info(group, bands):
    params = [f"LSST_obs_{b}" for b in bands]
    params += ["redshift", "ra", "dec", "galaxy_id", "shear1", "shear2", "totalEllipticity1", "totalEllipticity2", "diskHalfLightRadiusArcsec", "spheroidHalfLightRadiusArcsec", "bulge_frac"]
    data = {p: group[p][:] for p in params}
    output = {}
    output["ra"] = data["ra"]
    output["dec"] = data["dec"]
    output["redshift_true"] = data["redshift"]
    output["g1"] = data["shear1"]
    output["g2"] = data["shear2"]
    output["e1"] = data["totalEllipticity1"]
    output["e2"] = data["totalEllipticity2"]
    output["id"] = data["galaxy_id"]
    for b in bands:
        output[f"mag_{b}"] = data[f'LSST_obs_{b}']

    # Compute the overall (bulge + disc) half-light radius.
    # This copies the GCRCatalogs approach, which
    # I'm noy sure is quite right.
    hlr_disc = data["diskHalfLightRadiusArcsec"]
    hlr_bulge = data["spheroidHalfLightRadiusArcsec"]
    f = data["bulge_frac"]
    hlr = (hlr_disc * (1 - f) + hlr_bulge * f)

    # Convert half-light radius to T
    output["T"] = half_light_radius_to_trace(hlr)
    return output
    

def half_light_radius_to_trace(hlr):
    """Convert half-light radius to trace T of the Gaussian matrix"""
    size_sigma = hlr / np.sqrt(2 * np.log(2))
    size_T = 2 * size_sigma**2
    return size_T
