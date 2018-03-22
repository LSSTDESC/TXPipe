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


    Methods
    -------
    ang2pix(ra, dec, radians=False, theta=False)
        Convert ra and dec angles to pixel indices. 

    pix2ang(pix, radians=False, theta=False)
        Convert pixel indices to ra and dec angles


    """
    def __init__(self, nside, nest=False):
        """Convert angular sky coordinates to pixel indices.

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

def choose_pixelization(**config):
    """Construct a pixelization scheme based on configuration choices.

    Currently only the Healpix pixelization is defined.

    If kwargs['pixelization']=='healpix' then these parameters will be checked:
    nside: int, required

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
    pixelization = config['pixelization']
    if pixelization == 'healpix':
        import healpy
        nside = config['nside']
        if not healpy.isnsideok(nside):
            raise ValueError(f"nside pixelization parameter must be set to a power of two (used value {nside})")
        nest = config.get('nest', False)
        scheme = HealpixScheme(nside, nest=nest)
    else:
        raise ValueError("Pixelization scheme unknown")

    return scheme
