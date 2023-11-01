from .base_stage import PipelineStage
from .utils import choose_pixelization
from .data_types import (
    HDFFile,
    ShearCatalog,
    TextFile,
    MapsFile,
    FileCollection,
    FiducialCosmology,
    TomographyCatalog,
)
import glob
import time
import numpy as np

class TXLogNormalGlass(PipelineStage):
    """
    Uses GLASS to generate a simulated catalog from lognormal fields
    GLASS citation: 
    https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..11T

    Contamination is applied to the density field by poission sampling 
    the density field at a higher rate than target density, then 
    removing objects with prob proportional to 1/input_weight
    This allows us to produce contaminated/ucontamined maps of the same 
    galaxy realisation
    """

    name = "TXLogNormalGlass"
    parallel = False
    inputs = [ 
        ("mask", MapsFile),
        ("lens_photoz_stack", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("input_lss_weight_maps", MapsFile),
    ]

    outputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog_unweighted", TomographyCatalog), 
        #TO DO: add shear maps to output
    ]

    config_options = {
        "num_dens": None,
        "zmin": 0.,
        "zmax": 2.0,
        "dx": 100,
        "bias": [1.],
        "shift": 1.0,
        "contaminate": False,
        "random_seed": "def", 
    }

    def run(self):
        """
        Execute the TXLogNormalGlass pipeline stage

        This method coordinates the execution of the GLASS-based pipeline,
        generating simulated catalogs from lognormal fields

        It calls the following sub-methods:
        - set_z_shells
        - generate_cls
        - generate_catalogs
        - save_catalogs
        """

        #get nside and nest info from mask
        with self.open_input("mask", wrapper=True) as map_file:
            mask_map_info = map_file.read_map_info("mask")
        self.nside = self.lmax = mask_map_info["nside"]
        self.nest = mask_map_info["nest"]

        self.set_z_shells()

        self.generate_cls()

        self.generate_catalogs()

        self.save_catalogs()

    def set_z_shells(self):
        """
        Set up redshift shells with uniform spacing in comoving distance

        These shells determine the redshift resolution of the simulation so should
        be narrower than the n(z)s

        This method initializes redshift bins based on the provided configuration
        options. It calculates redshift shells and associated weight functions
        for later use

        Uses CCl rather than glass.shells.distance_grid to avoid mixing CCL and CAMB
        """
        import glass.shells

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        chi_min = cosmo.comoving_radial_distance(1/(1+self.config["zmin"]))
        chi_max = cosmo.comoving_radial_distance(1/(1+self.config["zmax"]))
        chi_grid = np.arange(chi_min, chi_max+self.config["dx"], self.config["dx"])
        a_grid = cosmo.scale_factor_of_chi(chi_grid)
        self.zb = (1./a_grid) - 1.
        self.ws = glass.shells.tophat_windows(self.zb, weight=camb_tophat_weight)
        self.nshells = len(self.ws)
        #TO DO: figure out why GLASS needed the linear ramp weight here

    def generate_cls(self):
        """
        Generate angular power spectra (C(l)s) for each redshift shell

        This method computes matter C(l)s using CCL for each pair of redshift shells based on the
        provided fiducial cosmology
        """
        import scipy.interpolate
        import pyccl as ccl

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        # In order for CCl to get the cross correlation between bins right, 
        # we need to interpolate the windows at all z
        dz = self.ws[0].za[1]-self.ws[0].za[0]
        zb_grid = np.arange(self.config["zmin"], self.config["zmax"]+dz, dz)

        #Make density shell objects for CCL
        density = []
        for ishell in range(self.nshells):
            bz = np.ones(len(zb_grid)) #bias should be 1 for matter shells
            wa_interped =  scipy.interpolate.interp1d( self.ws[ishell].za, self.ws[ishell].wa,
            bounds_error=False,fill_value=0. )(zb_grid)

            density.append( ccl.NumberCountsTracer(
                cosmo, 
                dndz=(zb_grid, wa_interped),
                has_rsd=False, 
                bias=(zb_grid,bz), 
                mag_bias=None
                )
            )

        self.ell = np.arange(self.lmax)
        self.cls = []
        self.cls2 = []
        self.cls_dict = {}
        for i in range(1,self.nshells+1):
            for j in range(i, 0, -1):
                cl_bin = cosmo.angular_cl(density[i-1], density[j-1], self.ell)
                self.cls.append( cl_bin )
                self.cls2.append( [(i,j),cl_bin] )
                self.cls_dict[i,j] = cl_bin

        print('Cls done')

    def generate_catalogs(self):
        """
        Generate simulated galaxy catalogs based on lognormal fields

        This method simulates galaxy positions within each matter shell and
        populates them with corresponding redshifts and bin information

        """
        import glass.fields
        import glass.points
        import glass.galaxies
        import healpy as hp 

        if self.config["random_seed"] == "def":
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(self.config["random_seed"]))

        #load n(z)
        nzs = []
        with self.open_input("lens_photoz_stack") as f:
            z_nz = f["n_of_z/lens/z"][:]
            bin_names = [k for k in f['n_of_z/lens'].keys() if 'bin_' in k]
            for bin_name in bin_names:
                nzs.append(f[f"n_of_z/lens/"+bin_name][:])

        #load mask
        with self.open_input("mask", wrapper=True) as map_file:
            mask = map_file.read_map("mask")
            self.mask_map_info = map_file.read_map_info("mask")
        mask[mask==hp.UNSEEN] = 0. #set UNSEEN pixels to 0 (GLASS needs this)
        mask_area = np.sum(mask[mask!=hp.UNSEEN])*hp.nside2pixarea(self.nside,degrees=True)

        #get number density arcmin^-2 for each z bin from config
        target_num_dens = np.array(self.config["num_dens"])
        assert len(target_num_dens) == len(nzs)
        self.nbin_lens = len(nzs)

        #prepare the Lognormal C(l)
        self.gls = glass.fields.lognormal_gls(self.cls, shift=self.config["shift"], nside=self.nside, lmax=self.lmax, ncorr=3)

        #check for negative values in the gls
        for i,g in enumerate(self.gls):
            if len(g) > 0:
                if (g<0).any():
                    print('negative values found in gC(l)',i,'setting to 0.')
                    print(g[g<0])
                    g[g<0] = 0.

        # generator for lognormal matter fields
        matter = glass.fields.generate_lognormal(self.gls, self.nside, shift=self.config["shift"], ncorr=3, rng=rng )

        #prepare weight maps
        #apply_contamination = self.ping_input_weight()
        if self.config["contaminate"]:
            max_inv_w = self.get_max_inverse_weight()

        # simulate and add galaxies in each matter shell 
        shell_catalogs = []
        for ishell, delta_i in enumerate(matter):
            print('computing shell', ishell)

            # restrict galaxy distributions to this shell 
            z_i, dndz_i = glass.shells.restrict(z_nz, nzs, self.ws[ishell])
            if (dndz_i==0).all():
                continue

            # compute galaxy density (for each n(z)) in this shell
            ngal_in_shell = target_num_dens*np.trapz(dndz_i, z_i)/np.trapz(nzs, z_nz)
            if self.config["contaminate"]:
                ngal_in_shell *= max_inv_w*ngal_in_shell
            print('Ngal',ngal_in_shell*mask_area*60*60)

            # simulate positions from matter density
            # TO DO: add galaxy biasing
            for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(ngal_in_shell, delta_i, bias=self.config["bias"], vis=mask, rng=rng ):

                #Figure out which bin was generated (len(ngal_in_shell) = Nbins)
                occupied_bins = np.where(gal_count != 0)[0]
                assert len(occupied_bins) == 1 #only one bin should be generated per call
                ibin = occupied_bins[0]
                gal_count_bin = gal_count[ibin]

                # sample redshifts uniformly in shell
                gal_z = glass.galaxies.redshifts_from_nz(gal_count_bin, self.ws[ishell].za, self.ws[ishell].wa)

                gal_lon[gal_lon < 0] += 360 #keep 0 < ra < 360

                if self.config["contaminate"]:
                    obj_pixel = hp.ang2pix(self.mask_map_info['nside'], gal_lon, gal_lat, lonlat=True, nest=True)
                    obj_weight = self.get_obj_weight(ibin, obj_pixel)
                    prob_accept = (1./obj_weight)/max_inv_w[ibin]
                    obj_accept_contaminated = np.random.rand(len(gal_lon)) < prob_accept
                    gal_lon = gal_lon[obj_accept_contaminated]
                    gal_lat = gal_lat[obj_accept_contaminated]
                    gal_z   = gal_z[obj_accept_contaminated]
                    gal_count_bin = np.sum(obj_accept_contaminated.astype('int'))

                shell_catalog = np.empty( gal_count_bin, 
                    dtype=[('RA', float), ('DEC', float), ('Z_TRUE', float), ('BIN', int)] )
                shell_catalog['RA'] = gal_lon
                shell_catalog['DEC'] = gal_lat
                shell_catalog['Z_TRUE'] = gal_z
                shell_catalog['BIN'] = np.full(gal_count_bin, ibin)
                shell_catalogs.append(shell_catalog)

                # TO DO: add a save_catalog_chunk method to save to hdf5 file in-loop 
                # (look at lens_selector.TXBaseLensSelector for an example of how to do this)

        self.catalog = np.hstack(shell_catalogs)
        self.counts = np.bincount(self.catalog['BIN'])
        del shell_catalogs

    def save_catalogs(self):
        """
        Save generated catalogs to output files

        This method saves the generated photometry catalog and lens tomography
        catalog to output files. The catalogs are stored in HDF5 format

        Note: It currently saves RA, DEC and Z_TRUE in the photometry catalog and bin
        information in the lens tomography catalog
        """

        #TO DO: use iterator for the catalogs in case they get big
        # e.g. look at LensNumberDensityStats in utils/number_density_stats.py

        #save the RA and DEC in a "photometry catalog"
        phot_output = self.open_output("photometry_catalog", parallel=True)
        group = phot_output.create_group("photometry")
        group.create_dataset("ra", data=self.catalog["RA"], dtype="f")
        group.create_dataset("dec", data=self.catalog["DEC"], dtype="f")
        group.create_dataset("redshift_true", data=self.catalog["Z_TRUE"], dtype="f")
        phot_output.close()

        tomo_output = self.open_output("lens_tomography_catalog_unweighted", parallel=True)
        group = tomo_output.create_group("tomography")
        group.create_dataset("bin", data=self.catalog['BIN'], dtype="i")
        group.create_dataset("lens_weight", data=np.ones(len(self.catalog)), dtype="f")
        group.create_dataset("counts", data=self.counts, dtype="i")
        group.create_dataset("counts_2d", data = np.array([np.sum(self.counts)]), dtype="i")
        
        group.attrs["nbin"] = self.nbin_lens
        tomo_output.close()

    def get_max_inverse_weight(self):
        """
        Get the maximum 1/weight for each tomographic bin
        """
        max_inv_w = np.ones(self.nbin_lens)
        with self.open_input("input_lss_weight_maps") as f:
            for tomobin in range(self.nbin_lens):
                value = f[f'maps/weight_map_bin_{tomobin}/value'][:]
                max_inv_w[tomobin] = (1./value).max()
            assert f['maps'].attrs['nside'] == self.mask_map_info['nside']
        return max_inv_w

    def get_obj_weight(self, tomobin, obj_pix):
        """
        Returns the weight for each object in a given tomographic bin
        """
        with self.open_input("input_lss_weight_maps", wrapper=True) as f:
            wmap = f.read_map(f"weight_map_bin_{tomobin}")
        return wmap[obj_pix]


def camb_tophat_weight(z):
    """
    TAKEN FROM GLASS

    This function returns a weight for a given redshift (z) based on a linear
    ramp from 0 to 1, transitioning from z=0 to z=0.1.

    Args:
        z (float): The redshift at which to calculate the weight.

    Returns:
        float: The weight for the given redshift, between 0 and 1.
    """
    return np.clip(z/0.1, None, 1.)



