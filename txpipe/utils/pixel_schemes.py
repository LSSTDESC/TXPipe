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

    @classmethod
    def read_map(HealpixScheme,fname_map,i_map=0) :
        #TODO: need to write a write_maps function
        """Read a map from a fits file and generate the associated HealpixScheme.

        Parameters
        ----------

        fname_map: string
            Path to file
        i_map: None, int or array-like
            Maps to read. If None, all maps are read.
        """
        import healpy as hp
        maps,h=hp.read_map(fname_map,field=i_map,h=True,verbose=False)
        h=dict(h);

        #Determine parameters
        nest=False ;
        if h['ORDERING']=='NESTED' :
            nest=True

        nside=h['NSIDE']

        p=HealpixScheme(nside,nest=nest)
        return p,maps

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
    pad: int
        Number of additional pixels around the edges set to zero.

    Methods
    -------
    ang2pix(ra, dec, radians=False, theta=False)
        Convert ra and dec angles to pixel indices. 

    pix2ang(pix, radians=False, theta=False)
        Convert pixel indices to ra and dec angles


    """    
    def __init__(self, ra_cent, dec_cent, pixel_size, nx, ny, pad=0,
                 pixel_size_y=None,crpix_x=None,crpix_y=None):
        """Make a converter object

        Parameters
        ----------
        ra_cent: float
            Right ascension at the patch centre
        dec_cent: float
            Declination at the patch centre
        pixel_size: float
            The size of each pixel along the side, in degrees
        nx, ny: int
            Number of pixels in the x and y directions
        pad:
            Number of additional empty pixels to leave around the edges
        pixel_size_y: float
            Pixel size in the y direction. If None, x and y will use the same pixel size.
            Otherwise, pixel_size will be interpreted as the size in the x direction.
        crpix_x: float
            Pixel index in the x direction for the reference point. If None, it will
            default to nx/2
        crpix_y: float
            Pixel index in the y direction for the reference point. If None, it will
            default to ny/2
        """
        from astropy.wcs import WCS

        if pixel_size_y is None :
            pixel_size_y=pixel_size
        if crpix_x is None :
            crpix_x=nx/2
        if crpix_y is None :
            crpix_y=ny/2

        wcs = WCS(naxis=2)
        #Note: we're assuming we'll want the tangent point to be at (nx/2,ny/2).
        #      This is common, but not universal.
        wcs.wcs.crpix = [crpix_x+pad, crpix_y+pad]
        wcs.wcs.cdelt = [pixel_size, pixel_size_y]
        wcs.wcs.crval = [ra_cent,dec_cent] #Pick middle point
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        self.wcs = wcs
        self.pixel_size_x=pixel_size
        self.pixel_size_y=pixel_size_y
        self.ra_cent = ra_cent
        self.dec_cent = dec_cent
        self.nx = nx
        self.ny = ny
        self.npix = self.nx*self.ny
        self.pad = pad

        self.metadata = {
            'ra_cent':self.ra_cent, 'dec_cent':self.dec_cent, 
            'pixel_size_x':pixel_size, 'pixel_size_y':pixel_size_y,
            'nx': self.nx, 'ny':self.ny, 'npix': self.npix,'pad': self.pad
            }


    def ang2pix_real(self, ra, dec, radians=False, theta=False):
        """Convert angular sky coordinates to tangent Cartesian coordinates.

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
        ix,iy: array
            Flat-sky coordinates in pixel units.
        """
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        if radians:
            ra = np.degrees(ra)
            dec = np.degrees(dec)
        if theta:
            dec = 90.0 - dec
        x, y = self.wcs.wcs_world2pix(ra, dec, 0)

        return x, y

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
            Flat-sky pixel indices of all the specified angles.
        """
        x, y = self.ang2pix_real(ra,dec,radians=radians,theta=theta)
        x = np.floor(x).astype(int)
        y = np.floor(y).astype(int)
        bad = (x<0) | (x>=self.nx) | (y<0) | (y>=self.ny)
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

        if theta:
            dec = 90.0 - dec
        if radians:
            ra = np.radians(ra)
            dec = np.radians(dec)
        return ra, dec

    @classmethod
    def read_map(GnominicPixelScheme,fname_map,i_map=0) :
        #TODO: need to write a write_maps function
        """Read a flat-sky map from a fits file and generate the associated
        GnomonicPixelScheme.

        Parameters
        ----------

        fname_map: string
            Path to file
        i_map: None, int or array-like
            Maps to read. If None, all maps are read.
        """
        from astropy.io import fits
        from astropy.wcs import WCS

        hdul=fits.open(fname_map)
        w=WCS(hdul[0].header)
        p=GnomonicPixelScheme(w.wcs.crval[0],w.wcs.crval[0],w.wcs.cdelt[0],
                              hdul[0].header['NAXIS1'],hdul[0].header['NAXIS2'],
                              pad=0,pixel_size_y=w.wcs.cdelt[1],
                              crpix_x=w.wcs.crpix[0],crpix_y=w.wcs.crpix[1])

        scalar_input=False
        if i_map is None :
            lst=np.arange(len(hdul))
        elif isinstance(i_map,(list,tuple,np.ndarray)) :
            lst=i_map
        else :
            scalar_input=True
            lst=np.array([i_map])

        maps=np.array([hdul[i].data for i in lst])
        if scalar_input :
            maps=maps.flatten()
        else :
            nm,ny,nx=maps.shape
            maps=maps.reshape([nm,ny*nx])
            
        return p,maps

                                                    
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
            raise ValueError("nside pixelization parameter must be set to a power of two (used value {nside})")
        nest = config.get('nest', False)
        scheme = HealpixScheme(nside, nest=nest)
    elif pixelization == 'gnomonic' or pixelization == 'tan' or pixelization == 'tangent':
        ra_cent = config['ra_cent']
        dec_cent = config['dec_cent']
        nx = config['npix_x']
        ny = config['npix_y']
        pixel_size = config['pixel_size']
        if np.isnan([ra_cent, dec_cent, nx, ny, pixel_size]).any():
            raise ValueError("Must set ra_cent, dec_cent, nx, ny, pixel_size to use Gnomonic/Tangent pixelization")
        scheme = GnomonicPixelScheme(ra_cent, dec_cent, pixel_size, nx, ny)
    else:
        raise ValueError("Pixelization scheme unknown")

    return scheme
