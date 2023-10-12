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
    """

    name = "TXLogNormalGlass"
    parallel = False
    inputs = [ 
        ("mask", MapsFile),
        ("lens_photoz_stack", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]

    outputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog_unweighted", TomographyCatalog), 
        #TO DO: add shear maps to output
    ]

    config_options = {
        "num_dens": None,
        "zmin":0.,
        "zmax":2.0,
        "dx":100,
    }

    def run(self):

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
        Redshift grid with uniform spacing in comoving distance
        uses CCl rather than glass.shells.distance_grid to avoid mixing CCL and CAMB
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
        Generate C(l)s for each shell at the fiducial cosmology 
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
        """
        import glass.fields
        import glass.points
        import glass.galaxies
        import healpy as hp 

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
            mask_map_info = map_file.read_map_info("mask")
        #mask[mask == hp.UNSEEN] = 0.0 #glass doesn't take UNSEEN values
        #mask_area = mask_map_info['area']
        mask_area = np.sum(mask[mask!=hp.UNSEEN])*hp.nside2pixarea(self.nside,degrees=True)

        #get number density arcmin^-2 for each z bin from config
        num_dens = np.array(self.config["num_dens"])
        assert len(num_dens) == len(nzs)
        self.nbin_lens = len(nzs)

        #prepare the Lognormal C(l)
        self.gls = glass.fields.lognormal_gls(self.cls, nside=self.nside, lmax=self.lmax, ncorr=3)

        #check for negative values in the gls
        for i,g in enumerate(self.gls):
            if len(g) > 0:
                if (g<0).any():
                    print('negative values found in gC(l)',i,'setting to 0.')
                    print(g[g<0])
                    g[g<0] = 0.

        # generator for lognormal matter fields
        matter = glass.fields.generate_lognormal(self.gls, self.nside, ncorr=3)

        # simulate and add galaxies in each matter shell 
        shell_catalogs = []
        for ishell, delta_i in enumerate(matter):
            print('computing shell', ishell)

            # restrict galaxy distributions to this shell 
            z_i, dndz_i = glass.shells.restrict(z_nz, nzs, self.ws[ishell])
            if (dndz_i==0).all():
                continue

            # compute galaxy density (for each n(z)) in this shell
            ngal_in_shell = num_dens*np.trapz(dndz_i, z_i)/np.trapz(nzs, z_nz)
            print('Ngal',ngal_in_shell*mask_area*60*60)


            # simulate positions from matter density
            # TO DO: add galaxy biasing
            for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(ngal_in_shell, delta_i, vis=mask):

                #Figure out which bin was generated
                occupied_bins = np.where(gal_count != 0)[0]
                assert len(occupied_bins) == 1 #only one bin should be generated per call
                ibin = occupied_bins[0]
                gal_count_bin = gal_count[ibin]

                # sample redshifts uniformly in shell
                gal_z = glass.galaxies.redshifts_from_nz(gal_count_bin, self.ws[ishell].za, self.ws[ishell].wa)

                gal_lon[gal_lon < 0] += 360 #keep 0 < ra < 360

                shell_catalog = np.empty( gal_count_bin, 
                    dtype=[('RA', float), ('DEC', float), ('Z_TRUE', float), ('BIN', int)] )
                shell_catalog['RA'] = gal_lon
                shell_catalog['DEC'] = gal_lat
                shell_catalog['Z_TRUE'] = gal_z
                shell_catalog['BIN'] = np.full(gal_count_bin, ibin)
                shell_catalogs.append(shell_catalog)

        self.catalog = np.hstack(shell_catalogs)
        del shell_catalogs

    def save_catalogs(self):
        """
        """

        #TO DO: use iterator for the catalogs in case they get big
        # e.g. look at LensNumberDensityStats in utils/number_density_stats.py

        #save the RA and DEC in a "photometry catalog"
        phot_output = self.open_output("photometry_catalog", parallel=True)
        group = phot_output.create_group("photometry")
        group.create_dataset("ra", data=self.catalog["RA"], dtype="f")
        group.create_dataset("dec", data=self.catalog["DEC"], dtype="f")
        phot_output.close()

        tomo_output = self.open_output("lens_tomography_catalog_unweighted", parallel=True)
        group = tomo_output.create_group("tomography")
        group.create_dataset("bin", data=self.catalog['BIN'], dtype="i")
        group.create_dataset("lens_weight", data=np.ones(len(self.catalog)), dtype="f")
        #group.create_dataset("counts", (nbin_lens,), dtype="i")
        #group.create_dataset("counts_2d", (1,), dtype="i")
        
        group.attrs["nbin"] = self.nbin_lens
        tomo_output.close()


def camb_tophat_weight(z):
    '''FROM GLASS
    Weight function for tophat window functions and CAMB.

    This weight function linearly ramps up the redshift at low values,
    from :math:`w(z = 0) = 0` to :math:`w(z = 0.1) = 1`.

    '''
    return np.clip(z/0.1, None, 1.)



