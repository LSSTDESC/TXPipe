from .base_stage import PipelineStage
from .data_types import MapsFile, YamlFile, RandomsCatalog, TomographyCatalog, HDFFile
from .utils import choose_pixelization
import numpy as np


class TXRandomCat(PipelineStage):
    name='TXRandomCat'
    inputs = [
        ('aux_maps', MapsFile),
        ('tracer_metadata', HDFFile),       
        ('lens_photoz_stack', HDFFile),
    ]
    outputs = [
        ('random_cats', RandomsCatalog),
    ]
    config_options = {
        'density': 100.,  # number per square arcmin at median depth depth.  Not sure if this is right.
        'Mstar': 23.0,  # Schecther distribution Mstar parameter
        'alpha': -1.25,  # Schecther distribution alpha parameter
        'sigma_e': 0.27,
    }

    def run(self):
        import scipy.special
        import scipy.stats
        import healpy
        from . import randoms
        # Load the input depth map
        with self.open_input('aux_maps', wrapper=True) as maps_file:
            depth = maps_file.read_map('depth/depth')
            info = maps_file.read_map_info('depth/depth')
            nside = info['nside']
            scheme = choose_pixelization(**info)

        pz_stack = self.open_input('lens_photoz_stack')

        # Cut down to pixels that have any objects in
        pixel = np.where(depth > 0)[0]
        depth = depth[pixel]

        if len(pixel)==1:
            raise ValueError("Only one pixel in depth map!")


        ### This will likely need to be changed with the density Schechter function and density
        ### being calculated individually for each redshift bin.  When it comes time to update the code
        ### to do that, move the lines below (down to the multi hash line) into the Ntomo loop

        # Read configuration values
        Mstar = self.config['Mstar']
        alpha15 = 1.5 + self.config['alpha']
        density_at_median = self.config['density']

        sigma_e = self.read_sigma_e()

        # Work out the normalization of a Schechter distribution
        # with the given median depth
        median_depth = np.median(depth)
        x_med = 10.**(0.4*(Mstar-median_depth))
        phi_star = density_at_median / scipy.special.gammaincc(alpha15, x_med)

        # Work out the number density in each pixel based on the 
        # given Schecter distribution
        x = 10.**(0.4*(Mstar-depth))
        density = phi_star * scipy.special.gammaincc(alpha15, x)

        # Pixel geometry - area in arcmin^2
        area = scheme.pixel_area(degrees=True) * 60. * 60.
        vertices = scheme.vertices(pixel)

        ##################################################################################

        ### I think this is redundant if your pz_stack file has separated lens bins

        ### The current file in 'pz_stack' only has 1 lens bin, but zbins loads 4 bins!!
        Ntomo = len(pz_stack['n_of_z']['lens'].keys())-1
        z_photo_arr = pz_stack['n_of_z']['lens']['z'][:]
        
        ### Loop over the tomographic bins to find number of galaxies in each pixel/zbin
        ### When the density changes per redshift bin, this can go into the main Ntomo loop
        numbers = {}
        for j in range(Ntomo):
            # Poisson distribution about mean
            numbers[j] = scipy.stats.poisson.rvs(density*area, 1)

        ### Get total number of randoms in all zbins
        ### Once the density gets updated per redshift bin, the output file will need to 
        ### combine all the tomographic bins in a bit more clever/convenient way than currently
        n_total = sum(numbers.values()).sum()

        output_file = self.open_output('random_cats')
        group = output_file.create_group('randoms')
        ra_out = group.create_dataset('ra', (n_total,), dtype=np.float64)
        dec_out = group.create_dataset('dec', (n_total,), dtype=np.float64)
        e1_out = group.create_dataset('e1', (n_total,), dtype=np.float64)
        e2_out = group.create_dataset('e2', (n_total,), dtype=np.float64)
        z_out = group.create_dataset('z', (n_total,), dtype=np.float64)
        bin_out = group.create_dataset('bin', (n_total,), dtype=np.float64)

        ### Counter for total number of randoms
        index = 0
        for j in range(Ntomo):
            print(f"Simulating tomographic bin {j}")
            ### Load pdf of ith lens redshift bin pz
            n_hist = pz_stack[f'n_of_z/lens/bin_{j}'][:]

            ### Make cdf and normalise
            z_cdf = np.cumsum(n_hist)
            z_cdf_norm = z_cdf / np.float(max(z_cdf))

            # Generate the random points in each pixel
            for i,(vertices_i,N) in enumerate(zip(vertices,numbers[j])):
                # First generate some random ellipticities.
                # This theta is not the orientation angle, it is the 
                # angle in the e1,e2 plane
                e = np.random.normal(scale=sigma_e[j], size=N)
                theta = np.random.uniform(0,2*np.pi,size=N)
                e1 = e * np.cos(theta)
                e2 = e * np.sin(theta)

                # Use the pixel vertices to generate the points
                ### This likely wont work for curved sky maps since healpy pixels aren't 
                ### fully quadrilateral... not sure how big of a difference (if any) this
                ### will make
                p1, p2, p3, p4 = vertices_i.T
                P = randoms.random_points_in_quadrilateral(p1, p2, p3, p4, N)
                # Convert to RA/Dec
                # This is not healpy-dependent so we just use it as a convenience function
                ra, dec = healpy.vec2ang(P, lonlat=True)

                bin_index = np.repeat(j, N)

                ### Create random values [0,1] equal to the number of galaxies per pixel
                cdf_rand_val = np.random.uniform(0,1.0,N)
                ### Interpolate those random values to a redshift value given by the cdf
                # z_photo_rand = np.interp(cdf_rand_val,z_cdf_norm,z_photo_arr)
                z_interp_func = scipy.interpolate.interp1d(z_cdf_norm,z_photo_arr)
                # Sometimes we don't quite go down to z - deal with that
                cdf_rand_val = cdf_rand_val.clip(z_cdf_norm.min(), z_cdf_norm.max())
                z_photo_rand = z_interp_func(cdf_rand_val)
                
                # Save output
                ra_out[index:index+N] = ra
                dec_out[index:index+N] = dec
                e1_out[index:index+N] = e1
                e2_out[index:index+N] = e2
                z_out[index:index+N] = z_photo_rand
                bin_out[index:index+N] = bin_index
                index += N

        output_file.close()

    def read_sigma_e(self):
        meta = self.open_input('tracer_metadata')
        d = meta['tracers/sigma_e'][:]
        meta.close()
        return d




if __name__ == '__main__':
    PipelineStage.main()
