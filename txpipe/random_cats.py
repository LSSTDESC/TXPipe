from ceci import PipelineStage
from descformats.tx import DiagnosticMaps, YamlFile, RandomsCatalog
import numpy as np


class TXRandomCat(PipelineStage):
    name='TXRandomCat'
    inputs = [
        ('diagnostic_maps', DiagnosticMaps),
        ('config', YamlFile),
    ]
    outputs = [
        ('random_cats', RandomsCatalog),
    ]
    config_options = {
        'density': 100.,  # number per square arcmin at median depth depth.  Not sure if this is right.
        'chunk_rows': 10000,  # number per square arcmin at median depth depth.  Not sure if this is right.
        'Mstar': 23.0,  # Schecther distribution Mstar parameter
        'alpha': -1.25,  # Schecther distribution Mstar parameter
        'sigma_e': 0.27,
    }

    def run(self):
        import scipy.special
        import scipy.stats
        import healpy
        from . import randoms
        # Load the input depth map
        maps_file = self.open_input('diagnostic_maps')
        pixel = maps_file['maps/depth/pixel'][:]
        depth = maps_file['maps/depth/value'][:]
        nside = maps_file['maps/depth'].attrs['nside']


        # Read configuration values
        Mstar = self.config['Mstar']
        alpha15 = 1.5 + self.config['alpha']
        density_at_median = self.config['density']
        # NEED TO GET SIGMA_E AS A MAP!
        # BUT THEN ALSO NEED TO AVOID INCLUDING THE ACTUAL LENSING IN IT.
        # WORK NEEDED!
        sigma_e = self.config['sigma_e']

        # Work out the normalization of a Schechter distribution
        # with the given median depth
        median_depth = np.median(depth)
        x_med = 10.**(0.4*(Mstar-median_depth))
        phi_star = density_at_median / scipy.special.gammaincc(alpha15, x_med)

        # Work out the number density in each pixel based on the 
        # given Schecter distribution
        x = 10.**(0.4*(Mstar-depth))
        density = phi_star * scipy.special.gammaincc(alpha15, x)

        # Pixel geometry
        boundaries = healpy.boundaries(nside, pixel)
        area = healpy.nside2pixarea(nside, degrees=True) * 60.*60.

        # Poisson distribution about mean
        numbers = scipy.stats.poisson.rvs(density*area, 1)
        n_total = numbers.sum()

        output_file = self.open_output('random_cats')
        group = output_file.create_group('randoms')
        ra_out = group.create_dataset('ra', (n_total,), dtype=np.float64)
        dec_out = group.create_dataset('dec', (n_total,), dtype=np.float64)
        e1_out = group.create_dataset('e1', (n_total,), dtype=np.float64)
        e2_out = group.create_dataset('e2', (n_total,), dtype=np.float64)

        index = 0
        # Generate the random points in each pixel
        for i,(vertices,N) in enumerate(zip(boundaries,numbers)):
            # First generate some random ellipticities.
            # This theta is not the orientation angle, it is the 
            # angle in the e1,e2 plane
            e = np.random.normal(scale=sigma_e, size=N)
            theta = np.random.uniform(0,2*np.pi,size=N)
            e1 = e * np.cos(theta)
            e2 = e * np.sin(theta)

            # Use the pixel vertices to generate the points
            p1, p2, p3, p4 = vertices.T
            P = randoms.random_points_in_quadrilateral(p1, p2, p3, p4, N)
            # Convert to RA/Dec
            ra, dec = healpy.vec2ang(P, lonlat=True)
            # Save output
            ra_out[index:index+N] = ra
            dec_out[index:index+N] = dec
            e1_out[index:index+N] = e1
            e2_out[index:index+N] = e2
            index += N

        output_file.close()
        


if __name__ == '__main__':
    PipelineStage.main()
