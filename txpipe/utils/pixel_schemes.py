import numpy as np

class HealpixScheme:
    """A pixelization scheme using Healpix pixels.

    Attributes
    ----------
    nside: int
        The Healpix resolution parameter
    npix: int
        The maximum pixel index.
    nest:
        Whether the map is in the nested pixel ordering scheme, as opposed to the ring scheme
    metadata: dict
        Dictionary of metadata describing the projection


    Methods
    -------
    ang2pix(ra, dec, radians=False, theta=False)
        Convert ra and dec angles to pixel indices. 

    pix2ang(pix, radians=False, theta=False)
        Convert pixel indices to ra and dec angles


    """
    def __init__(self, nside, nest=False):
        """Make a converter object.

        Parameters
        ----------

        nside: int
            The Healpix resolution parameter
        nest: bool, optional
            Whether to use the nested pixel ordering scheme.  Default=False
        """
        import healpy
        self.healpy = healpy
        self.nside = nside
        self.nest = nest
        self.npix = self.healpy.nside2npix(self.nside)

        self.metadata = {'nside':self.nside, 'npix':self.npix, 'nest':self.nest}

    def ang2pix(self, ra, dec, radians=False, theta=False):
        """Convert angular sky coordinates to pixel indices.

        Parameters
        ----------

        ra: array or float
            Right ascension coordinates, in degrees unless radians=True
        dec: array or float
            Declination coordinates, in degrees unless radians=True
        radians: bool, optional
            if True, assume the input values are in radians.  
        theta: bool, optional
            If True, assume the input "dec" values are co-latitude theta (90-declination, the "angle from north").

        Returns
        -------
        pix: array
            Healpix indices of all the specified angles.
        """
        if not radians:
            ra = np.radians(ra)
            dec = np.radians(dec)
        if not theta:
            dec = np.pi/2. - dec
        return self.healpy.ang2pix(self.nside, dec, ra, nest=self.nest)

    def pix2ang(self, pix, radians=False, theta=False):
        """Convert pixel indices to angular sky coordinates.

        Parameters
        ----------

        pix: array
            Healpix indices to convert to angles
        radians: bool, optional
            if True, return output in radians not degrees
        theta: bool, optional
            If True, return the co-latitude theta (90-declination, the "angle from north").

        Returns
        -------
        ra: array or float
            Right ascension coordinates, in degrees unless radians=True

        dec: array or float
            Declination coordinates, in degrees unless radians=True.
            If theta=True instead return the co-latitude theta (90-declination, the "angle from north").


        """
        theta, phi = self.healpy.ang2pix(self.nside, pix, nest=self.nest)
        if not theta:
            theta = np.pi/2 - theta
        if not radians:
            theta = np.degrees(theta)
            phi = np.degrees(phi)
        return phi, theta


class GnomonicPixelScheme:
    """A pixelization scheme using the Gnomonic (aka tangent plane) projection.

    The pixel index is of the form p = x + y*nx

    Attributes
    ----------
    ra_min: float
        The minimum right ascension value to use in the map, in degrees
    ra_max: float
        The maximum right ascension value to use in the map, in degrees
    dec_min: float
        The minimum declination value to use in the map, in degrees
    dec_max: float
        The maximum declination value to use in the map, in degrees
    pixel_size: float
        The size of each pixel along the side, in degrees
    metadata: dict
        Dictionary of metadata describing the projection
    npix: int
        Total number of pixels
    nx: int
        Number of pixels along x direction
    ny: int
        Number of pixels along y direction

    Methods
    -------
    ang2pix(ra, dec, radians=False, theta=False)
        Convert ra and dec angles to pixel indices. 

    pix2ang(pix, radians=False, theta=False)
        Convert pixel indices to ra and dec angles


    """    
    def __init__(self, ra_min, ra_max, dec_min, dec_max, pixel_size):
        """Make a converter object

    Parameters
    ----------
    ra_min: float
        The minimum right ascension value to use in the map, in degrees
    ra_max: int
        The maximum right ascension value to use in the map, in degrees
    dec_min:
        The minimum declination value to use in the map, in degrees
    dec_max:
        The maximum declination value to use in the map, in degrees
    pixel_size:
        The size of each pixel along the side, in degrees



        """
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [0., 0.]
        wcs.wcs.cdelt = [pixel_size, pixel_size]
        wcs.wcs.crval = [ra_min, dec_min]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        self.wcs = wcs
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.nx = int(np.ceil((ra_max - ra_min) / pixel_size))
        self.ny = int(np.ceil((dec_max - dec_min) / pixel_size))
        self.npix = self.nx*self.ny

        self.metadata = {
            'ra_min':self.ra_min, 'ra_max':self.ra_max, 
            'dec_min':self.dec_min, 'dec_max':self.dec_max,
            'nx': self.nx, 'ny':self.ny,
            'npix': self.npix
            }



    def ang2pix(self, ra, dec, radians=False, theta=False):
        """Convert angular sky coordinates to pixel indices.

        Parameters
        ----------

        ra: array or float
            Right ascension coordinates, in degrees unless radians=True
        dec: array or float
            Declination coordinates, in degrees unless radians=True
        radians: bool, optional
            if True, assume the input values are in radians.  
        theta: bool, optional
            If True, assume the input "dec" values are co-latitude theta (90-declination, the "angle from north").

        Returns
        -------
        pix: array
            Healpix indices of all the specified angles.
        """
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        if radians:
            ra = np.degrees(ra)
            dec = np.degrees(dec)
        if theta:
            dec = 90.0 - dec
        x, y = self.wcs.wcs_world2pix(ra, dec, 1)
        x = np.floor(x).astype(int)
        y = np.floor(y).astype(int)
        bad = (ra<=self.ra_min) | (dec<=self.dec_min) | (ra>=self.ra_max) | (dec>=self.dec_max)
        pix = x + y*self.nx
        pix[bad] = -9999
        return pix

    def pix2ang(self, pix, radians=False, theta=False):
        """Convert pixel indices to angular sky coordinates.

        Parameters
        ----------

        pix: array
            Indices to convert to angles
        radians: bool, optional
            if True, return output in radians not degrees
        theta: bool, optional
            If True, return the co-latitude theta (90-declination, the "angle from north").

        Returns
        -------
        ra: array or float
            Right ascension coordinates, in degrees unless radians=True

        dec: array or float
            Declination coordinates, in degrees unless radians=True.
            If theta=True instead return the co-latitude theta (90-declination, the "angle from north").
        """
        pix = np.atleast_1d(pix)
        x = pix % self.nx
        y = pix // self.nx
        ra, dec = self.wcs.wcs_pix2world(x, y, 0.0)
        bad = (pix<0) | (pix>=self.npix)

        if theta:
            dec = 90.0 - dec
        if radians:
            ra = np.radians(ra)
            dec = np.radians(dec)
        ra[bad] = np.nan
        dec[bad] = np.nan
        return ra, dec


def choose_pixelization(**config):
    """Construct a pixelization scheme based on configuration choices.

    Currently only the Healpix pixelization is defined.

    If kwargs['pixelization']=='healpix' then these parameters will be checked:

    Parameters
    ----------
    pixelization: str
        Choice of pixelization, currently only "healpix" is supported
    nside: int, optional
        Only used if pixelization=='healpix'.  Healpix resolution parameter, must be power of 2.
    nest: bool, optional
        Only used if pixelization=='healpix'.  Healpix ordering scheme. Default=False

    Returns
    -------
    scheme: PixelizationScheme
        Instance of a pixelization scheme subclass

    """
    pixelization = config['pixelization'].lower()
    if pixelization == 'healpix':
        import healpy
        nside = config['nside']
        if not healpy.isnsideok(nside):
            raise ValueError(f"nside pixelization parameter must be set to a power of two (used value {nside})")
        nest = config.get('nest', False)
        scheme = HealpixScheme(nside, nest=nest)
    elif pixelization == 'gnomonic' or pixelization == 'tan' or pixelization == 'tangent':
        ra_min = config['ra_min']
        dec_min = config['dec_min']
        ra_max = config['ra_max']
        dec_max = config['dec_max']
        pixel_size = config['pixel_size']
        scheme = GnomonicPixelScheme(ra_min, ra_max, dec_min, dec_max, pixel_size)
    else:
        raise ValueError("Pixelization scheme unknown")

    return scheme
