"""
This file contains TXPipe-specific file types, subclassing the more
generic types in base.py
"""

from .base import HDFFile, DataFile, YamlFile
from ..mapping import degrade_healsparse
import yaml
import numpy as np
import warnings

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

    def get_true_redshift_column(self):
        if self.catalog_type == "metadetect":
            return "00/redshift_true"
        else:
            return "redshift_true"

    def get_primary_catalog_names(self, true_shear=False):
        if true_shear:
            if self.catalog_type == "metadetect":
                shear_cols = ["00/true_g1", "00/true_g2", "00/ra", "00/dec", "00/weight"]
                rename = {c: c[3:] for c in shear_cols}
                rename["00/true_g1"] = "g1"
                rename["00/true_g2"] = "g2"
            else:
                rename = {"true_g1": "g1", "true_g2": "g2"}
                rename = {}
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

    def get_bands(self, shear_prefix=""):
        group = self.file[self.get_primary_catalog_group()]
        if "bands" in group.attrs:
            return group.attrs["bands"]

        # If we don't have it listed correctly then we have to do this
        # messy check. We look for all the columns that start with
        # mag_ and don't have an extra underscore in, which would indicate
        # that they are an error like mag_z_err
        bands = []
        nunderscore = shear_prefix.count("_")
        l = len(shear_prefix + "mag_")
        for col in group.keys():
            if col.startswith(f"{shear_prefix}mag_") and (col.count("_") == 1 + nunderscore):
                bands.append(col[l:])
        return bands


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
        zbins = [(d[f"zmin_{i}"], d[f"zmax_{i}"]) for i in range(nbin)]
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
                # legacy maps have datasets "pixel", "value"
                # healsparse maps have sub-group "healsparse"
                # so if one of these is there then this will
                # be a map
                if ("pixel" in keys and "value" in keys) or ("healsparse" in keys):
                    maps.append(name)

        # Now actually run this
        self.file["maps"].visititems(visit)

        # return the accumulated list
        return maps
    
    def read_healsparse(self, map_name, **kwargs):
        """
        Read healsparse map from hdf5
        
        All keyword arguments are forwarded to HealSparseMap.read
        """
        import healsparse as hsp
        return hsp.HealSparseMap.read(self.path, hdf5_group=f"maps/{map_name}/healsparse", **kwargs)

    def check_is_legacy(self, map_name):
        """
        Returns True if map was saved using an old TXPipe file format
        i.e. is not a healsparse file
        """
        keys = self.file[f"maps/{map_name}"].keys()
        return ("pixel" in keys) and ("value" in keys)

    def read_healpix_legacy(self, map_name):
        import healsparse as hsp
        import healpy

        group = self.file[f"maps/{map_name}"]
        nside = group.attrs["nside"]
        m = hsp.HealSparseMap.make_empty(32, nside, dtype=type(group["value"][0]))
        if group.attrs['nest']:
            pix = group["pixel"][:]
        else:
            pix = healpy.ring2nest(nside, group["pixel"][:])
        m.update_values_pix(pix, group["value"][:])
        return m

    def read_map_info(self, map_name):
        group = self.file[f"maps/{map_name}"]
        info = dict(group.attrs)
        if not "pixelization" in info:
            warnings.warn(
                f"Warning 'pixelization' not found in '{map_name}' in file {self.path}.\n"
                "This might mean this is not a TXPipe file.\n"
                "We will default to assuming a healpix map"
                , UserWarning)
            info["pixelization"] = "healpix"
        return info

    def read_map(self, map_name):
        """
        Read map and return as a healsparse map
        
        Parameters
        ----------

        map_name: `str`
            The name of this map
        """
        info = self.read_map_info(map_name)
        pixelization = info["pixelization"]
        if pixelization == "gnomonic":
            m = self.read_gnomonic(map_name)
        elif pixelization == "healpix":
            is_legacy = self.check_is_legacy(map_name)

            if is_legacy:
                m = self.read_healpix_legacy(map_name)
            else:
                m = self.read_healsparse(map_name)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m

    def read_map_healpix(self, map_name, nside=None, reduction='mean', key=None, nest=True):
        """
        Read map and return as a healpix array
        
        Parameters
        ----------

        map_name: `str`
            The name of this map
        nside : `int` (healsparse argument)
            Output nside resolution parameter (should be a multiple of 2). If
            not specified the output resolution will be equal to the parent's
            sparsemap nside_sparse
        reduction : `str` (healsparse argument)
            If a change in resolution is requested, this controls the method to
            reduce the map computing the "mean", "median", "std", "max", "min",
            "sum" or "prod" (product)  of the neighboring pixels to compute the
            "degraded" map.
        key : `str` (healsparse argument)
            If the parent HealSparseMap contains recarrays, key selects the
            field that will be transformed into a HEALPix map.
        nest : `bool`, optional (healsparse argument)
            Output healpix map should be in nest format?
        """
        hsp_map = self.read_map(map_name)
        m = hsp_map.generate_healpix_map(nside=nside, reduction=reduction, key=key, nest=nest)
        return m

    def read_mask(self, mask_name=None, thresh=0, degrade_nside=None, returnbool=False):
        """
        Read the mask and return as a healsparse map

        Parameters
        ----------

        map_name: str or None  (optional)
            The name of this mask, if None wil load the default "mask"
        thresh: float (optional)
            minimum fractional coverage of a pixel (at native nside)
        degrade_nside: int
            if required, degrade the mask to this nside
        returnbool: bool
            if True, will return a binary map where any pixel with mask > 0 is True
            if mask was already boolean, return unchanged
        """
        import healsparse as hsp

        if mask_name is None:
            mask_name = "mask"
        mask = self.read_map(mask_name)

        #remove any pixels below the threshold
        pix_to_cut = mask.valid_pixels[mask[mask.valid_pixels]<=thresh]
        mask.update_values_pix(pix_to_cut, mask.sentinel)

        #degrade if requested and nessesary
        if (degrade_nside is not None) and (degrade_nside != mask.nside_sparse):
            mask = degrade_healsparse(mask, reduction="mask", degrade_nside=degrade_nside)

        if returnbool and not np.issubdtype(mask.dtype, np.bool_):
            #make a boolean mask from a frac map
            bool_mask = hsp.HealSparseMap.make_empty(mask.nside_coverage, mask.nside_sparse, bool)
            bool_mask[mask.valid_pixels] = True
            mask = bool_mask 

        return mask
    
    def read_mask_healpix(self, mask_name=None, thresh=0., degrade_nside=None):
        """
        Read the mask and return as a healpix array

        Parameters
        ----------

        map_name: str or None  (optional)
            The name of this mask, if None wil load the default "mask"
        thresh: float (optional)
            minimum fractional coverage of a pixel (at native nside)
        degrade_nside : int or None  (optional)
            degrade the mask to this nside before converting to healpix array 
        """
        import healsparse as hsp
        mask_out = self.read_mask(mask_name=mask_name, thresh=thresh, degrade_nside=degrade_nside)
        return mask_out.generate_healpix_map()

    def write_map(self, map_name, hsp_map, metadata):
        """
        Save an output healsparse map to an HDF5 subgroup in healsparse format

        Parameters
        ----------

        map_name: str
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
        
        hsp_map.write(self.path, format='hdf5', clobber=True, hdf5_group=f"maps/{map_name}/healsparse")

        #also add any extra metadata
        subgroup = self.file[f"maps/{map_name}"]
        subgroup.attrs.update(metadata)

    def write_map_pixval(self, map_name, pixel, value, metadata, nside_coverage=32):
        """
        Save an array of pixel indicies and values to an HDF5 subgroup in healsparse format

        The metadata is also saved.

        Parameters
        ----------

        map_name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        nside_coverage: int
            nside of the healsparse coverage map
        """
        import healsparse as hsp
        import healpy 

        if not "maps" in self.file:
            self.file.create_group("maps")
        if not "pixelization" in metadata:
            raise ValueError("Map metadata should include pixelization")
        pixerr = ValueError(f"Map pixels and values should be same shape but are {pixel.shape} vs {value.shape}")
        if value.ndim == 2:  # value.ndim == 2 allows for 2d stacked maps
            if not pixel.shape[0] == value.shape[1]:
                raise pixerr
        else:
            if not pixel.shape == value.shape:
                raise pixerr

        subgroup = self.file["maps"].create_group(map_name)
        subgroup.attrs.update(metadata)

        #convert pixels and values into a healsparse map and save
        if value.ndim == 2: #input values are a stack of maps. save as recarray (labels 0, 1, 2 etc)
            dtypes = [(str(i), value.dtype) for i in range(value.shape[0])]
            value = np.rec.fromarrays(value, dtype=dtypes)
            hsp_map = hsp.HealSparseMap.make_empty(
                        nside_coverage=nside_coverage,
                        nside_sparse=metadata['nside'],
                        dtype=dtypes,
                        primary='0')
        else:
            hsp_map = hsp.HealSparseMap.make_empty(
                        nside_coverage=nside_coverage,
                        nside_sparse=metadata['nside'],
                        dtype=value.dtype)
        if metadata['nest']:
            hsp_pix = pixel
        else:
            hsp_pix = healpy.ring2nest(metadata['nside'], pixel)
        
        hsp_map.update_values_pix(hsp_pix, value)

        hsp_map.write(self.path, format='hdf5', clobber=True, hdf5_group=f"maps/{map_name}/healsparse")

    def plot_healpix(self, map_name, view="cart", rot180=False, nside=None, reduction='mean', key=None, weight_map=None, **kwargs):
        """
        Plots healpix map using healpy tools

        The map is read as a HealSparse map, optionally degraded 
        to a target nside, converted to a Healpix array, and plotted with healpy

        Parameters
        ----------
        map_name : str
            Name of the map to read and plot.
        view : {"cart", "moll"}, optional
            Healpy view type: Cartesian ("cart") or Mollweide ("moll").
        rot180 : bool, optional
            If True, rotate the map by 180 degrees in longitude before plotting.
        nside : int, optional
            Target Healpix nside for visualization. Defaults to the sparse
            nside of the input map.
        reduction : str, optional
            Reduction operation used when generating the Healpix map
            from the HealSparse representation (e.g. "mean", "sum").
        key : str, optional
            Optional key used if healsparse map is a recarray
        **kwargs
            Additional keyword arguments passed directly to the underlying
            healpy plotting function (e.g. ``min``, ``max``, ``cmap``).
        """
        import healpy
        import numpy as np

        info = self.read_map_info(map_name)
        assert info["pixelization"] != "gnomonic"

        hsp_map = self.read_map(map_name)

        if nside is None:
            nside = hsp_map.nside_sparse
        pix = hsp_map.valid_pixels
        if reduction in ["weightedmean", "mask"]: #custom degrading reductions
            degraded_map = degrade_healsparse(hsp_map, nside, reduction, weight_map)
            m = degraded_map.generate_healpix_map()
        else:
            m = hsp_map.generate_healpix_map(nside=nside, reduction=reduction, key=key)
        
        lon, lat = healpy.pix2ang(nside, pix, lonlat=True)
        if rot180:  # (optional) rotate 180 degrees in the lon direction
            lon += 180
            lon[lon > 360.0] -= 360.0
            pix_rot = healpy.ang2pix(nside, lon, lat, lonlat=True)
            m_rot = np.ones(healpy.nside2npix(nside)) * healpy.UNSEEN
            m_rot[pix_rot] = m[pix]
            m = m_rot
            pix = pix_rot
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
        lon_range = np.clip(lon_range, 0, 360.0)
        m[m == 0] = healpy.UNSEEN
        title = kwargs.pop("title", map_name)
        if view == "cart":
            healpy.cartview(m, lonra=lon_range, latra=lat_range, title=title, hold=True, nest=info["nest"], **kwargs)
        elif view == "moll":
            healpy.mollview(m, title=title, hold=True, nest=info["nest"], **kwargs)
        else:
            raise ValueError(f"Unknown Healpix view mode {view}")

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

    def read_rotation_healpix(self, realization_index, bin_index):
        g1,g2 = self.read_rotation(realization_index, bin_index)
        g1_hp = g1.generate_healpix_map()
        g2_hp = g2.generate_healpix_map()
        return g1_hp, g2_hp

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

    def read_density_split_healpix(self, realization_index, bin_index):
        rho1,rho2 = self.read_density_split(realization_index, bin_index)
        rho1_hp = rho1.generate_healpix_map()
        rho2_hp = rho2.generate_healpix_map()
        return rho1_hp, rho2_hp

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
            raise ValueError("Do not use the open_output method to write sacc files.  Use sacc.write_fits")
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
            inits.update(
                dict(
                    bcm_log10Mc=params["bcm_log10Mc"],
                    bcm_etab=params["bcm_etab"],
                    bcm_ks=params["bcm_ks"],
                    mu_0=params["mu_0"],
                    sigma_0=params["sigma_0"],
                )
            )

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
        try:
            read = tables_io.io.readHdf5GroupToDict
        except AttributeError:
            read = tables_io.hdf5.read_HDF5_group_to_dict
        meta = read(self.file["meta"])
        self._metadata = meta
        return meta


class QPPDFFile(QPBaseFile):
    def iterate(self, chunk_rows, rank=0, size=1):
        import qp

        return qp.iterator(self.path, chunk_size=chunk_rows, rank=rank, parallel_size=size)

    def get_z(self):
        import qp

        metadata = qp.read_metadata(self.path)
        return metadata["xvals"].copy().squeeze()

    def get_pdf_type(self):
        import qp

        metadata = qp.read_metadata(self.path)
        return metadata["pdf_name"][0].decode()


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

        # Use cached ensemble if available
        if self._ensemble is not None:
            return self._ensemble

        # Build ensemble, following approach in qp factory code
        try:
            read = tables_io.io.readHdf5GroupToDict
        except AttributeError:
            read = tables_io.hdf5.read_HDF5_group_to_dict
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
            raise ValueError(
                f"Requested n(z) for bin {bin_index} but only {npdf - 1} bins available. For the 2D bin use get_2d_n_of_z."
            )
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
        try:
            read = tables_io.io.readHdf5GroupToDict
        except AttributeError:
            read = tables_io.hdf5.read_HDF5_group_to_dict
        return read(self.file[f"qps/{name}/meta"])

    def read_ensemble(self, name):
        import qp
        import tables_io

        g = self.file[f"qps/{name}"]
        try:
            read = tables_io.io.readHdf5GroupToDict
        except AttributeError:
            read = tables_io.hdf5.read_HDF5_group_to_dict
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
