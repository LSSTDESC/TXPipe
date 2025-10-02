import numpy as np
from ...base_stage import PipelineStage
from ...data_types import MapsFile, TextFile


class TXIngestPlanckLensingMaps(PipelineStage):
    """Ingest Planck NPIPE CMB lensing maps.

    This stage reads in the Planck PR4 CMB lensing maps, and mask, and
    noise spectrum.

    Input files:
    - data/planck/PR4_klm_dat_p.fits: The Planck PR4 CMB lensing alms.
    - data/planck/mask.fits.gz: The Planck PR4 CMB lensing mask.
    - data/planck/PR4_klm_dat_p_noise.fits: The Planck PR4 CMB lensing noise spectrum.
    """

    name = "TXIngestPlanckLensingMaps"
    inputs = []
    outputs = [
        ("cmb_lensing_map", MapsFile)
    ]
    config_options = {
        "alm_file": "data/planck/PR4_klm_dat_p.fits",
        "mask_file": "data/planck/mask.fits.gz",
        "noise_file": "data/planck/PR4_klm_dat_p_noise.fits",
        "nside": 512,
    }

    def run(self):
        kappa_map = self.ingest_kappa()
        mask = self.ingest_mask()

        kappa_pix = np.where(mask > 0)[0]
        kappa_val = kappa_map[kappa_pix]
        mask_val = mask[kappa_pix]

        noise_ell, noise_spectrum = self.ingest_noise_spectrum()

        with self.open_output("cmb_lensing_map", wrapper=True) as f:
            f.write_map("kappa_cmb", kappa_pix, kappa_val)
            f.write_map("kappa_mask", kappa_pix, mask_val)

            # add a separate section for the noise
            g = f.file.create_group("noise_spectrum")
            g.create_dataset("noise_n_ell", data=noise_ell)
            g.create_dataset("noise_spectrum", data=noise_spectrum)


    def ingest_kappa(self):
        import healpy as hp
        alm_file = self.config['alm_file']
        nside = self.config['nside']

        # Read in the CMB lensing alms
        alms, lmax = hp.read_alm(alm_file, return_mmax=True)

        # correct for the monopole
        alms[0] = 0 + 0j

        # Rotate from Galactic to Celestial coordinates
        rot = hp.Rotator(coord=['G', 'C'])
        alms = rot.rotate_alm(alms)

        # Low-pass filter: remove power above ell = 3 * Nside
        fl = np.ones(lmax+1)
        fl[3*nside:] = 0
        alms = hp.almxfl(alms, fl, inplace=True)

        # convert to map space
        kappa = hp.alm2map(alms, nside)
        return kappa

    def ingest_mask(self):
        import healpy as hp
        import pymaster as nmt
        mask_file = self.config['mask_file']
        nside = self.config['nside']

        # Read the Planck lensing mask
        mask = hp.read_map(mask_file)

        # Rotate from Galactic to Celestial coordinates
        rot = hp.Rotator(coord=['G', 'C'])
        mask = rot.rotate_map_pixel(mask)

        # Apodize. The size is the scale in degrees.
        # The type means the definition listed here:
        # https://namaster.readthedocs.io/en/latest/api/pymaster.utils.html#pymaster.utils.mask_apodization
        mask = nmt.mask_apodization(mask, aposize=0.2, apotype="C1")

        # Convert to a binary mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # Downgrade to desired nside and return
        mask = hp.ud_grade(mask, nside_out=nside)
        return mask
    
    def ingest_noise_spectrum(self):
        noise_file = self.config['noise_file']
        noise_spectrum = np.loadtxt(noise_file)
        ell = np.arange(len(noise_spectrum))
        return ell, noise_spectrum



class TXIngestQuaia(PipelineStage):
    """Ingest Quaia lensing maps as TXPipe overdensity (delta) maps."""

    config_options = {
        "quaia_file": "data/quaia_G20.5.fits",
        "selection_function_template": "data/selection_function_NSIDE64_G20.5_zsplit2bin{}.fits",
        "nside": 512,
        "sel_threshold": 0.5,
        "num_z_bins": 500,
        "zname": "redshift_quaia",
    }

    def run(self):
        import healpy as hp
        from astropy.table import Table
        

        nside = self.config['nside']
        npix = hp.nside2npix(nside)
        zname = self.config['zname']
        quaia_file = self.config['quaia_file']

        cat = Table.read(quaia_file)
        cat.sort(zname)
        zs = cat[zname]

        # The quasar catalog is sorted by redshift already so the
        # median redshift is just the middle element
        zmedian = zs[len(cat)//2]
        z_edges = np.array([0, zmedian, 5])

        # Low and high redshift bins
        cat1 = cat[(zs < z_edges[1]) & (zs >= z_edges[0])]
        cat2 = cat[(zs < z_edges[2]) & (zs >= z_edges[1])]

        print("Redshift edges: ", z_edges)
        cats = [cat1, cat2]

        sel_file_name = self.config['selection_function_template']
        sels = [
            hp.ud_grade(hp.read_map(sel_file_name.format(i)), nside_out=nside) 
            for i in range(2)
        ]
        
        maps = {}
        for i in range(2):
            pix, delta = self.process_catalog(cats[i], sels[i])
            maps[f'delta_{i}'] = (pix, delta)

        with self.open_output("density_maps", wrapper=True) as f:
            for name, (pix, delta) in maps.items():
                f.write_map(name, pix, delta)

    def process_catalog(self, cat, sel):
        import healpy as hp
        import pymaster as nmt

        sel_threshold = self.config['sel_threshold']
        nside = self.config['nside']
        npix = hp.nside2npix(nside)
        num_z_bins = self.config['num_z_bins']
        zname = self.config['zname']

        # Threshold selection function
        mask_b = sel > sel_threshold
        mask = sel.copy()
        mask[~mask_b] = 0.0

        # Get angular mask and cut catalog
        ipix = hp.ang2pix(nside, cat['ra'], cat['dec'], lonlat=True)
        maskflag = mask_b[ipix]
        c = cat[maskflag]
        ipix = ipix[maskflag]

        # Get redshift distribution via PDF stacking
        zs = np.linspace(0., 4.5, num_z_bins)
        sz = c[zname+'_err']
        zm = c[zname]
        nz = np.array([np.sum(np.exp(-0.5*((z-zm)/sz)**2)/sz)
                    for z in zs])

        # Calculate overdensity field
        nmap = np.bincount(ipix, minlength=npix)
        nmean = np.sum(nmap*mask_b)/np.sum(mask*mask_b)
        delta = np.zeros(npix)

        delta[mask_b] = nmap[mask_b]/(nmean*mask[mask_b])-1

        return np.where(mask_b), delta


    maps = {}
    for i in range(2):
        maps[f'qso{i}'] = process_catalog(cats[i], sels[i])

