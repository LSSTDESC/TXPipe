import glob
import numpy as np
from ..data_types import ShearCatalog, TextFile, QPPDFFile, PhotometryCatalog
from ..base_stage import PipelineStage
from .mock_tools import add_lsst_like_noise, make_metadetect_catalog, TableWriter, ShearTableWriter, make_photo_cuts, half_light_radius_to_trace
from ceci.config import StageParameter


class TXIngestExtraGalactic(PipelineStage):
    name = "TXIngestExtraGalactic";
    inputs = []
    parallel = False
    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", PhotometryCatalog),
    ]

    config_options = {
        "delta_gamma": StageParameter(float, default=0.02, msg="Delta gamma value for metadetect response calculations"),
        "year": StageParameter(int, default=1, msg="Number of years of LSST observations to simulate photometric noise for"),
        "response_type": StageParameter(str, default="unit", msg="Type of response to apply for metadetect"),
        "snr_cut": StageParameter(float, default=5.0, msg="SNR cut for overall detection"),
        "T_ratio_cut": StageParameter(float, default=0.5, msg="T/PSF_T cut for metadetect catalog"),
        "random_seed": StageParameter(int, default=0, msg="Random seed"),
        "bands": StageParameter(str, default="ugrizy", msg="Bands to ingest photometry for"),
        "shear_bands": StageParameter(str, default="griz", msg="Bands to use for shear metadetect"),
    }    

    def run(self):
        shear_filename = self.get_output("shear_catalog")
        photo_file = self.open_output("photometry_catalog")
        photo_group = photo_file.create_group("photometry")

        snr_cut = self.config["snr_cut"]
        T_ratio_cut = self.config["T_ratio_cut"]
        rng = np.random.default_rng(self.config['random_seed'])
        size = self.config.get("initial_size", 10_000_000)
        
        with ShearTableWriter(shear_filename, initial_size=size) as shear_writer, \
            TableWriter(photo_group, initial_size=size) as photo_writer:
            for data in self.data_iterator():
                add_lsst_like_noise(data, rng, year=self.config["year"])
                shear_data = make_metadetect_catalog(data, self.config["response_type"], self.config["delta_gamma"], self.config["shear_bands"], rng, snr_cut=snr_cut, T_ratio_cut=T_ratio_cut)
                photo_data = make_photo_cuts(data, self.config["bands"], snr_cut)
                shear_writer.write(shear_data)
                photo_writer.write(photo_data)

    def data_iterator(self):
        raise NotImplementedError("data_iterator must be implemented in subclass")




class TXIngestRomanRubin(TXIngestExtraGalactic):
    """Ingest skysim simulation data for TXPipe processing.
    """
    name = "TXIngestRomanRubin"
    config_options = TXIngestExtraGalactic.config_options | {
        "xgal_dir_name": StageParameter(str, default="/global/cfs/cdirs/lsst/shared/xgal/roman-rubin/roman_rubin_2023_v1.1.3", msg="Directory name for skysim xgal files"),
        "file_pattern": StageParameter(str, default="roman_rubin_2023_*.hdf5", msg="Pattern for file name"),
    }

    def data_iterator(self):
        import h5py
        bands = "ugrizy"
        file_format = self.config["file_pattern"]
        files = glob.glob(f"{self.config.xgal_dir_name}/{file_format}")
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
                    data = self.extract_roman_rubin_truth_info(group, bands)
                    yield data

    def extract_roman_rubin_truth_info(self, group, bands):
        params = [f"LSST_obs_{b}" for b in bands]
        params += ["redshift", "ra", "dec", "galaxy_id", "shear1", "shear2",  "diskHalfLightRadiusArcsec", "spheroidHalfLightRadiusArcsec", "bulge_frac"]
        if "totalEllipticity1" in group:
            params += ["totalEllipticity1", "totalEllipticity2"]
        else:
            params += ["spheroidEllipticity1", "spheroidEllipticity2", "diskEllipticity1", "diskEllipticity2"]
        data = {p: group[p][:] for p in params}
        output = {}
        output["ra"] = data["ra"]
        output["dec"] = data["dec"]
        output["redshift_true"] = data["redshift"]
            
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

        if "totalEllipticity1" in data:
            e1 = data["totalEllipticity1"]
            e2 = data["totalEllipticity2"]
        else:
            e1 = data["spheroidEllipticity1"] * data["bulge_frac"] + data["diskEllipticity1"] * (1 - data["bulge_frac"])
            e2 = data["spheroidEllipticity2"] * data["bulge_frac"] + data["diskEllipticity2"] * (1 - data["bulge_frac"])

        g = data["shear1"] + 1j * data["shear2"]
        e = e1 + 1j * e2
        # Apply shear to get observed ellipticity
        denom = 1 + np.conj(g) * e
        e_obs = (e + g) / denom
        output["g1"] = np.real(e_obs)
        output["g2"] = np.imag(e_obs)
        output["true_g1"] = data["shear1"]
        output["true_g2"] = data["shear2"]


        # Convert half-light radius to T
        output["T"] = half_light_radius_to_trace(hlr)
        return output
            


class TXIngestSkySim(TXIngestExtraGalactic):
    """Ingest skysim simulation data for TXPipe processing.
    """
    name = "TXIngestSkySim"
    config_options = TXIngestExtraGalactic.config_options | {
        "cat_name": StageParameter(str, default="skysim5000_v1.2", msg="Name of the GCR catalog to use"),
        "extra_cols": StageParameter(str, default="", msg="Extra columns to ingest from the catalog"),
    }

    def data_iterator(self):
        import GCRCatalogs
        gc = GCRCatalogs.load_catalog(self.config["cat_name"])
        # Columns we need from the cosmo simulation
        cols = [
            "ra",
            "dec",
            "mag_true_u_lsst",
            "mag_true_g_lsst",
            "mag_true_r_lsst",
            "mag_true_i_lsst",
            "mag_true_z_lsst",
            "mag_true_y_lsst",
            "ellipticity_1_true",
            "ellipticity_2_true",
            "shear_1",
            "shear_2",
            "size_true",
            "galaxy_id",
            "redshift_true",
        ]
        # Add any extra requestd columns
        cols += self.config["extra_cols"].split()

        it = gc.get_quantities(cols, return_iterator=True)
        nfile = len(gc._file_list) if hasattr(gc, "_file_list") else 0

        for i, data in enumerate(it):
            if nfile:
                j = i + 1
                print(f"Loading chunk {j}/{nfile}")
            yield self.process_data(data)


    def process_data(self, data):
        output = {}

        # Basic info
        output["ra"] = data["ra"]
        output["dec"] = data["dec"]
        output["redshift_true"] = data["redshift_true"]
        output["id"] = data["galaxy_id"]

        # Magnitudes
        for b in "ugrizy":
            output[f"mag_{b}"] = data[f"mag_true_{b}_lsst"]

        # Shear and ellipticity info
        g = data["shear_1"] + 1j * data["shear_2"]
        e = data["ellipticity_1_true"] + 1j * data["ellipticity_2_true"]

        # Apply shear to get observed ellipticity
        denom = 1 + np.conj(g) * e
        e_obs = (e + g) / denom
        output["g1"] = np.real(e_obs)
        output["g2"] = np.imag(e_obs)
        output["true_g1"] = data["shear_1"]
        output["true_g2"] = data["shear_2"]

        size_hlr = data["size_true"]
        output["T"] = half_light_radius_to_trace(size_hlr)


class TXIngestCosmoDC2(TXIngestSkySim):
    """Ingest CosmoDC2 data for TXPipe processing.
    """
    name = "TXIngestCosmoDC2"
    config_options = TXIngestSkySim.config_options | {
        "cat_name": StageParameter(str, default="cosmoDC2", msg="Name of the GCR catalog to use"),
    }


class TXIngestBuzzard(TXIngestSkySim):
    """Ingest Buzzard data for TXPipe processing.
    """
    name = "TXIngestBuzzard"
    config_options = TXIngestSkySim.config_options | {
        "cat_name": StageParameter(str, default="buzzard", msg="Name of the GCR catalog to use"),
    }





class TXSimpleMock(PipelineStage):
    """
    Load an ascii astropy table and put it in shear catalog format.
    """

    name = "TXSimpleMock"
    parallel = False
    inputs = [("mock_shear_catalog", TextFile)]
    outputs = [("shear_catalog", ShearCatalog)]
    config_options = {}

    def run(self):
        from astropy.table import Table
        import numpy as np

        # Load the data. We are assuming here it is small enough to fit in memory
        input_filename = self.get_input("mock_shear_catalog")
        input_data = Table.read(input_filename, format="ascii")
        n = len(input_data)

        data = {}
        # required columns
        for col in ["ra", "dec", "g1", "g2", "s2n", "T"]:
            data[col] = input_data[col]

        # It's most likely we will have a redshift column.
        # Check for both that and "redshift_true"
        if "redshift" in input_data.colnames:
            data["redshift_true"] = input_data["redshift"]
        elif "redshift_true" in input_data.colnames:
            data["redshift_true"] = input_data["redshift_true"]

        # If there is an ID column then use it, but otherwise just use
        # sequential IDs
        if "id" in input_data.colnames:
            data["galaxy_id"] = input_data["id"]
        else:
            data["galaxy_id"] = np.arange(len(input_data))

        # if these catalogs are not present then we fake them.
        defaults = {
            "T_err": 0.0,
            "psf_g1": 0.0,
            "psf_g2": 0.0,
            "psf_T_mean": 0.202,  # this corresponds to a FWHM of 0.75 arcsec
            "weight": 1.0,
            "flags": 0,
        }

        for key, value in defaults.items():
            if key in input_data.colnames:
                data[key] = input_data[key]
            else:
                data[key] = np.full(n, value)

        self.save_catalog(data)

    def save_catalog(self, data):
        with self.open_output("shear_catalog") as f:
            g = f.create_group("shear")
            g.attrs["catalog_type"] = "simple"
            for key, value in data.items():
                g.create_dataset(key, data=value)


class TXMockTruthPZ(PipelineStage):
    name = "TXMockTruthPZ"
    parallel = False
    inputs = [("shear_catalog", ShearCatalog)]
    outputs = [("photoz_pdfs", QPPDFFile)]
    config_options = {
        "mock_sigma_z": StageParameter(float, 0.001, msg="Sigma_z for mock photo-z PDF generation."),
    }

    def run(self):
        import qp
        import numpy as np

        sigma_z = self.config["mock_sigma_z"]

        # read the input truth redshifts
        with self.open_input("shear_catalog", wrapper=True) as f:
            group = f.file[f.get_primary_catalog_group()]
            n = group["ra"].size
            redshifts = group["redshift_true"][:]

        zgrid = np.linspace(0, 3, 301)
        pdfs = np.zeros((n, len(zgrid)))

        spread_z = sigma_z * (1 + redshifts)
        # make a gaussian PDF for each object
        delta = zgrid[np.newaxis, :] - redshifts[:, np.newaxis]
        pdfs = np.exp(-0.5 * (delta / spread_z[:, np.newaxis]) ** 2) / np.sqrt(2 * np.pi) / spread_z[:, np.newaxis]

        q = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pdfs))
        q.set_ancil(dict(zmode=redshifts, zmean=redshifts, zmedian=redshifts))
        q.write_to(self.get_output("photoz_pdfs"))
