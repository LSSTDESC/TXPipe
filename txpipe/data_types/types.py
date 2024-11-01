"""
This file contains TXPipe-specific file types, subclassing the more
generic types in base.py
"""
from .base import HDFFile, DataFile, YamlFile
import yaml
import numpy as np

def metacalibration_names(names):
    """
    Generate the metacalibrated variants of the inputs names,
    that is, variants with _1p, _1m, _2p, and _2m on the end
    of each name.
    """
    suffices = ["1p", "1m", "2p", "2m"]
    out = []
    for name in names:
        out += [name + "_" + s for s in suffices]
    return out

class PhotometryCatalog(HDFFile):
    def get_bands(self):
        group = self.file["photometry"]
        if "bands" in group.attrs:
            return group.attrs["bands"]
        bands = []
        for col in group.keys():
            if col.startswith("mag_") and col.count("_") == 1:
                bands.append(col[4:])
        return bands


class ShearCatalog(HDFFile):
    """
    A generic shear catalog
    """

    # These are columns

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._catalog_type = None

    def read_catalog_info(self):
        try:
            group = self.file["shear"]
            info = dict(group.attrs)
        except:
            raise ValueError(f"Unable to read shear catalog")
        shear_catalog_type = info.get("catalog_type")
        return shear_catalog_type

    @classmethod
    def subclass_for_file(cls, path):
        with ShearCatalog(path, "r") as f:
            catalog_type = f.catalog_type
        if catalog_type == "metacal":
            return MetacalCatalog
        elif catalog_type == "metadetect":
            return MetadetectCatalog
        elif catalog_type == "lensfit":
            return LensfitCatalog
        elif catalog_type == "hsc":
            return HSCShearCatalog
        else:
            return GenericShearCatalog

    @property
    def catalog_type(self):
        if self._catalog_type is not None:
            return self._catalog_type

        if "catalog_type" in self.file["shear"].attrs:
            t = self.file["shear"].attrs["catalog_type"]
        elif "mcal_g1" in self.file["shear"].keys():
            t = "metacal"
        elif "1p" in self.file["shear"].keys():
            t = "metadetect"
        elif "c1" in self.file["shear"].keys():
            t = "lensfit"
        else:
            raise ValueError("Could not figure out catalog format")

        self._catalog_type = t
        return t

    def get_size(self):
        if self.catalog_type == "metadetect":
            return self.file["shear/00/ra"].size
        else:
            return self.file["shear/ra"].size

    def get_primary_catalog_group(self):
        if self.catalog_type == "metadetect":
            return "shear/00"
        else:
            return "shear"


    def get_primary_catalog_names(self, true_shear=False):
        if true_shear:
            shear_cols = ["true_g1", "true_g2", "ra", "dec", "weight"]
            rename = {"true_g1": "g1", "true_g2": "g2"}
        elif self.catalog_type == "metacal":
            shear_cols = ["mcal_g1", "mcal_g2", "ra", "dec", "weight"]
            rename = {"mcal_g1": "g1", "mcal_g2": "g2"}
        elif self.catalog_type == "hsc":
            shear_cols = ["g1", "g2", "c1", "c2", "ra", "dec", "weight"]
            rename = {}
        elif self.catalog_type == "metadetect":
            shear_cols = ["00/g1", "00/g2", "00/ra", "00/dec", "00/weight"]
            rename = {c: c[3:] for c in shear_cols}
        else:
            shear_cols = ["g1", "g2", "ra", "dec", "weight"]
            rename = {}

        return shear_cols, rename

    def get_bands(self):
        group = self.file[self.get_primary_catalog_group()]
        if "bands" in group.attrs:
            return group.attrs["bands"]
        bands = []
        for col in group.keys():
            if col.startswith("mag_") and col.count("_") == 1:
                bands.append(col[4:])
        return bands

class GenericShearCatalog(ShearCatalog):
    pass

class MetadetectCatalog(ShearCatalog):
    def get_size(self):
        return self.file["shear/00/ra"].size
    def get_primary_catalog_group(self):
        return "shear/00"

class LensfitCatalog(ShearCatalog):
    pass

class MetacalCatalog(ShearCatalog):
    pass

class HSCShearCatalog(ShearCatalog):
    pass

class BinnedCatalog(HDFFile):
    required_datasets = []
    def get_bins(self, group_name):
        group = self.file[group_name]
        info = dict(group.attrs)
        bins = []
        for i in range(info["nbin"]):
            code = info[f"bin_{i}"]
            name = f"bin_{code}"
            bins.append(name)
        return bins



class TomographyCatalog(HDFFile):
    required_datasets = []

    def write_zbins(self, edges):
        """
        Write redshift bin edges to attributes
        """
        d = self.file["tomography"].attrs
        d[f"nbin"] = len(edges) - 1
        for i, (zmin, zmax) in enumerate(zip(edges[:-1], edges[1:])):
            d[f"zmin_{i}"] = zmin
            d[f"zmax_{i}"] = zmax

    def read_zbins(self):
        """
        Read saved redshift bin edges from attributes
        """
        d = dict(self.file["tomography"].attrs)
        nbin = d[f"nbin"]
        zbins = [
            (d[f"zmin_{i}"], d[f"zmax_{i}"]) for i in range(nbin)
        ]
        return zbins
    
    def write_nbin(self, nbin):
        """
        Write number of redshift bins to attributes
        """
        d = dict(self.file["tomography"].attrs)
        d[f"nbin"] = nbin

    def read_nbin(self):
        d = dict(self.file["tomography"].attrs)
        return d[f"nbin"]


class RandomsCatalog(HDFFile):
    required_datasets = ["randoms/ra", "randoms/dec"]


class MapsFile(HDFFile):
    required_datasets = []

    def list_maps(self):
        import h5py

        maps = []

        # h5py uses this visititems method to walk through
        # a file, looking at everything underneath a path.
        # We use it here to search through everything in the
        # "maps" section of a maps file looking for any groups
        # that seem to be a map.  You have to pass a function
        # like this to visititems.
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                keys = obj.keys()
                # we save maps with these two data sets,
                # so if they are both there then this will
                # be a map
                if "pixel" in keys and "value" in keys:
                    maps.append(name)

        # Now actually run this
        self.file["maps"].visititems(visit)

        # return the accumulated list
        return maps

    def read_healpix(self, map_name, return_all=False):
        import healpy
        import numpy as np

        group = self.file[f"maps/{map_name}"]
        nside = group.attrs["nside"]
        npix = healpy.nside2npix(nside)
        m = np.repeat(healpy.UNSEEN, npix)
        pix = group["pixel"][:]
        val = group["value"][:]
        m[pix] = val
        if return_all:
            return m, pix, nside
        else:
            return m

    def read_map_info(self, map_name):
        group = self.file[f"maps/{map_name}"]
        info = dict(group.attrs)
        if not "pixelization" in info:
            raise ValueError(
                f"Map '{map_name}' not found, "
                f"or not saved properly in file {self.path}"
            )
        return info

    def read_map(self, map_name):
        info = self.read_map_info(map_name)
        pixelization = info["pixelization"]
        if pixelization == "gnomonic":
            m = self.read_gnomonic(map_name)
        elif pixelization == "healpix":
            m = self.read_healpix(map_name)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m

    def read_mask(self, thresh=0):
        mask = self.read_map("mask")
        mask[mask <= thresh] = 0
        return mask

    def write_map(self, map_name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup.

        The pixel numbering and the metadata are also saved.

        Parameters
        ----------

        group: H5Group
            The h5py Group object in which to store maps
        name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        """
        if not "maps" in self.file:
            self.file.create_group("maps")
        if not "pixelization" in metadata:
            raise ValueError("Map metadata should include pixelization")
        if not pixel.shape == value.shape:
            raise ValueError(
                f"Map pixels and values should be same shape "
                f"but are {pixel.shape} vs {value.shape}"
            )

        if not 'maps' in self.file.keys():
            self.file.create_group('maps')
        subgroup = self.file["maps"].create_group(map_name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)

    def plot_healpix(self, map_name, view="cart", **kwargs):
        import healpy
        import numpy as np

        m, pix, nside = self.read_healpix(map_name, return_all=True)
        lon, lat = healpy.pix2ang(nside, pix, lonlat=True)
        npix = healpy.nside2npix(nside)
        if len(pix) == 0:
            print(f"Empty map {map_name}")
            return
        if len(pix) == len(m):
            w = np.where((m != healpy.UNSEEN) & (m != 0))
        else:
            w = None
        lon_range = [lon[w].min() - 0.1, lon[w].max() + 0.1]
        lat_range = [lat[w].min() - 0.1, lat[w].max() + 0.1]
        lat_range = np.clip(lat_range, -90, 90)
        m[m == 0] = healpy.UNSEEN
        title = kwargs.pop("title", map_name)
        if view == "cart":
            healpy.cartview(
                m, lonra=lon_range, latra=lat_range, title=title, hold=True, **kwargs
            )
        elif view == "moll":
            healpy.mollview(m, title=title, hold=True, **kwargs)
        else:
            raise ValueError(f"Unknown Healpix view mode {mode}")

    def read_gnomonic(self, map_name):
        import numpy as np

        group = self.file[f"maps/{map_name}"]
        info = dict(group.attrs)
        nx = info["nx"]
        ny = info["ny"]
        m = np.zeros((ny, nx))
        m[:, :] = np.nan

        pix = group["pixel"][:]
        val = group["value"][:]
        w = np.where(pix != -9999)
        pix = pix[w]
        val = val[w]
        x = pix % nx
        y = pix // nx
        m[y, x] = val
        return m

    def plot_gnomonic(self, map_name, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np

        info = self.read_map_info(map_name)
        ra_min, ra_max = info["ra_min"], info["ra_max"]
        if ra_min > 180 and ra_max < 180:
            ra_min -= 360
        ra_range = (ra_max, ra_min)
        dec_range = (info["dec_min"], info["dec_max"])

        # the view arg is needed for healpix but not gnomonic
        kwargs.pop("view")
        m = self.read_gnomonic(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop("title", map_name)
        plt.imshow(m, aspect="equal", extent=extent, **kwargs)
        plt.title(title)
        plt.colorbar()

    def plot(self, map_name, **kwargs):
        info = self.read_map_info(map_name)
        pixelization = info["pixelization"]
        if pixelization == "gnomonic":
            m = self.plot_gnomonic(map_name, **kwargs)
        elif pixelization == "healpix":
            m = self.plot_healpix(map_name, **kwargs)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m


class LensingNoiseMaps(MapsFile):
    required_datasets = []

    def read_rotation(self, realization_index, bin_index):
        g1_name = f"rotation_{realization_index}/g1_{bin_index}"
        g2_name = f"rotation_{realization_index}/g2_{bin_index}"

        g1 = self.read_map(g1_name)
        g2 = self.read_map(g2_name)

        return g1, g2

    def number_of_realizations(self):
        info = self.file["maps"].attrs
        lensing_realizations = info["lensing_realizations"]
        return lensing_realizations


class ClusteringNoiseMaps(MapsFile):
    def read_density_split(self, realization_index, bin_index):
        rho1_name = f"split_{realization_index}/rho1_{bin_index}"
        rho2_name = f"split_{realization_index}/rho2_{bin_index}"
        rho1 = self.read_map(rho1_name)
        rho2 = self.read_map(rho2_name)
        return rho1, rho2

    def number_of_realizations(self):
        info = self.file["maps"].attrs
        clustering_realizations = info["clustering_realizations"]
        return clustering_realizations


class PhotozPDFFile(HDFFile):
    required_datasets = []

    def get_z_grid(self):
        return self.file["/meta/xvals"][:][0]


class CSVFile(DataFile):
    suffix = "csv"

    def save_file(self, name, dataframe):
        dataframe.to_csv(name)


class SACCFile(DataFile):
    suffix = "sacc"

    @classmethod
    def open(cls, path, mode, **kwargs):
        import sacc

        if mode == "w":
            raise ValueError(
                "Do not use the open_output method to write sacc files.  Use sacc.write_fits"
            )
        return sacc.Sacc.load_fits(path)

    def read_provenance(self):
        meta = self.file.metadata
        provenance = {
            "uuid": meta.get("provenance/uuid", "UNKNOWN"),
            "creation": meta.get("provenance/creation", "UNKNOWN"),
            "domain": meta.get("provenance/domain", "UNKNOWN"),
            "username": meta.get("provenance/username", "UNKNOWN"),
        }

        return provenance

    def close(self):
        pass


class FiducialCosmology(YamlFile):

    # TODO replace when CCL has more complete serialization tools.
    def to_ccl(self, **kwargs):
        import pyccl as ccl

        with open(self.path, "r") as fp:
            params = yaml.load(fp, Loader=yaml.Loader)

        # Now we assemble an init for the object since the CCL YAML has
        # extra info we don't need and different formatting.
        inits = dict(
            Omega_c=params["Omega_c"],
            Omega_b=params["Omega_b"],
            h=params["h"],
            n_s=params["n_s"],
            sigma8=None if params["sigma8"] == "nan" else params["sigma8"],
            A_s=None if params["A_s"] == "nan" else params["A_s"],
            Omega_k=params["Omega_k"],
            Neff=params["Neff"],
            w0=params["w0"],
            wa=params["wa"],
        )
        if ccl.__version__[0] == "2":
            inits.update(dict(
            bcm_log10Mc=params["bcm_log10Mc"],
            bcm_etab=params["bcm_etab"],
            bcm_ks=params["bcm_ks"],
            mu_0=params["mu_0"],
            sigma_0=params["sigma_0"],
            ))


        if "z_mg" in params:
            inits["z_mg"] = params["z_mg"]
            inits["df_mg"] = params["df_mg"]

        if "m_nu" in params:
            inits["m_nu"] = params["m_nu"]
            inits["m_nu_type"] = "list"

        inits.update(kwargs)

        return ccl.Cosmology(**inits)


class QPBaseFile(HDFFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "r":
            self._metadata = None

    @property
    def metadata(self):
        import tables_io
        if self._metadata is not None:
            return self._metadata
        meta = tables_io.io.readHdf5GroupToDict(self.file["meta"])
        self._metadata = meta
        return meta

class QPPDFFile(QPBaseFile):
    def iterate(self, chunk_rows, rank=0, size=1):
        import qp
        return qp.iterator(self.path, chunk_size=chunk_rows, rank=rank, parallel_size=size)

    def get_z(self):
        import qp
        metadata = qp.read_metadata(self.path)
        return metadata['xvals'].copy().squeeze()

    def get_pdf_type(self):
        import qp
        metadata = qp.read_metadata(self.path)
        return metadata['pdf_name'][0].decode()



class QPNOfZFile(QPBaseFile):
    """
    The final ensemble row represents the 2D (non-tomographic) n(z).

    In a few places TXPipe assumes that the pdf type is one of the
    grid types, and will raise an error otherwise; in particular the
    stacking stage.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensemble = None
    
    def write_ensemble(self, ensemble):
        if not "qp" in self.file.keys():
            self.file.create_group("qp")
        group = self.file["qp"]
        tables = ensemble.build_tables()
        for subgroup_name, subtables in tables.items():
            subgroup = group.create_group(subgroup_name)
            for key, val in subtables.items():
                subgroup.create_dataset(key, dtype=val.dtype, data=val.data)

    def read_ensemble(self):
        """
        Read the complete QP object from this file.
        """
        import qp
        import tables_io

        # Use cached ensemble if available
        if self._ensemble is not None:
            return self._ensemble

        # Build ensemble, following approach in qp factory code
        read = tables_io.io.readHdf5GroupToDict
        tables = dict([(key, read(val)) for key, val in self.file["qp"].items()])

        self._ensemble = qp.from_tables(tables)
        return self._ensemble
        
    def get_qp_pdf_type(self):
        meta = self.metadata
        pdf_name = meta["pdf_name"][0].decode()
        return pdf_name
    
    def get_nbin(self):
        ens = self.read_ensemble()
        return ens.npdf - 1
    
    def get_2d_n_of_z(self, zmax=3.0, nz=301):
        z = np.linspace(0, zmax, nz)
        ensemble = self.read_ensemble()
        return z, ensemble.pdf(z)[-1]

    def get_bin_n_of_z(self, bin_index, zmax=3.0, nz=301):
        ensemble = self.read_ensemble()
        npdf = ensemble.npdf
        if bin_index >= npdf - 1:
            raise ValueError(f"Requested n(z) for bin {bin_index} but only {npdf-1} bins available. For the 2D bin use get_2d_n_of_z.")
        z = np.linspace(0, zmax, nz)
        return z, ensemble.pdf(z)[bin_index]

    def get_z(self):
        """
        Get the redshift grid for this n(z) file.

        If the QP representation used does not have a simple z grid
        (e.g. if it is a gaussian mixture) then this will raise an error.
        """
        pdf_name = self.get_qp_pdf_type()
        meta = self.metadata
        if pdf_name == "interp":
            z = meta["xvals"][:]
        elif pdf_name == "hist":
            z = meta["bins"][:]
        else:
            raise ValueError(f"TXPipe cannot read a z grid from QP file with type {pdf_name}")
        return z.squeeze()
    
    
class QPMultiFile(HDFFile):
    """
    This type represents and HDF file collecting multiple qp objects together.

    We currently use it when multiple realizations of the same n(z) are
    being generated in the summarize stage.
    """
    def get_names(self):
        return list(self.file["qps"].keys())

    def read_metadata(self, name):
        import tables_io
        if self.mode != "r":
            raise ValueError("Can only read from file opened in read mode")
        return tables_io.io.readHdf5GroupToDict(self.file[f"qps/{name}/meta"])

    def read_ensemble(self, name):
        import qp
        import tables_io
        g = self.file[f"qps/{name}"]
        read = tables_io.io.readHdf5GroupToDict
        tables = dict([(key, read(val)) for key, val in g.items()])
        return qp.from_tables(tables)

    def write_ensemble(self, ensemble, name):
        if "qps" in self.file.keys():
            g = self.file["qps"]
        else:
            g = self.file.create_group("qps")
        group = g.create_group(name)
        
        tables = ensemble.build_tables()
        for subgroup_name, subtables in tables.items():
            subgroup = group.create_group(subgroup_name)
            for key, val in subtables.items():
                subgroup.create_dataset(key, dtype=val.dtype, data=val.data)
