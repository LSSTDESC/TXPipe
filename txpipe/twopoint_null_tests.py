from .base_stage import PipelineStage
from .data_types import HDFFile, ShearCatalog, TomographyCatalog, RandomsCatalog, SACCFile, PNGFile, TextFile
import numpy as np
from .twopoint import TXTwoPoint


class TXGammaTFieldCenters(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of exposure fields as "lenses", as a systematics test.
    """
    name = "TXGammaTFieldCenters"
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('exposures', HDFFile),
        ('patch_centers', TextFile),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('gammat_field_center', SACCFile),
        ('gammat_field_center_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_method': 'shot',
        'npatch': 5,
        'use_true_shear': False,
        'subtract_mean_shear':False
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        filename = self.get_input('exposures')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('exposures')
        data['lens_ra']  = f['exposures/ratel'][:]
        data['lens_dec'] = f['exposures/dectel'][:]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_field_center_plot', wrapper=True)

        plt.errorbar(dtheta,  dtheta*dvalue, derror, fmt='ro', capsize=3)
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Field Center Tangential Shear")

        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyFieldCenter_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'fieldcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'fieldcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        #self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_field_center'), overwrite=True)

        # Also make a plot of the data

class TXGammaTBrightStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTBrightStars"
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile),
        ('patch_centers', TextFile),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('gammat_bright_stars', SACCFile),
        ('gammat_bright_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_method': 'shot',
        'npatch': 5,
        'use_true_shear': False,
        'subtract_mean_shear': False,
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')

        mags = f['stars/mag_r'][:]
        bright_cut = mags>14
        bright_cut &= mags<18.3

        data['lens_ra']  = f['stars/ra'][:][bright_cut]
        data['lens_dec'] = f['stars/dec'][:][bright_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_bright_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        z = (dvalue) / derror
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(dtheta) - 1)
        print('error,',derror)

        plt.errorbar(dtheta,  dtheta*dvalue, dtheta*derror, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Bright Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_bright_stars'), overwrite=True)

        # Also make a plot of the data

class TXGammaTDimStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTDimStars"
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile),
        ('patch_centers', TextFile),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('gammat_dim_stars', SACCFile),
        ('gammat_dim_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_method': 'shot',
        'npatch': 5,
        'use_true_shear': False,
        'subtract_mean_shear': False,        
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')
        mags = f['stars/mag_r'][:]
        dim_cut = mags>18.2
        dim_cut &= mags<22

        data['lens_ra']  = f['stars/ra'][:][dim_cut]
        data['lens_dec'] = f['stars/dec'][:][dim_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_dim_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (dvalue - flat1) / derror
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(dtheta) - 1)
        print('error,',derror)

        plt.errorbar(dtheta,  dtheta*dvalue, dtheta*derror, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Dim Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight


        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_dim_stars'), overwrite=True)

        # Also make a plot of the data

class TXGammaTRandoms(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTRandoms"
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile),
        ('patch_centers', TextFile),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('gammat_randoms', SACCFile),
        ('gammat_randoms_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_method': 'shot',
        'npatch': 5,
        'use_true_shear': False,
        'subtract_mean_shear': False,
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_random_catalog(self, data):
        # override the parent method
        # so that we don't load the randoms here,
        # because if we subtract randoms from randoms
        # we get nothing.
        pass

    def load_lens_catalog(self, data):
        # We load the randoms to use as lenses
        f = self.open_input('random_cats')
        group = f['randoms']
        data['lens_ra'] = group['ra'][:]
        data['lens_dec'] = group['dec'][:]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_randoms_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (dvalue - flat1) / derror
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(dtheta) - 1)
        print('error,',derror)

        plt.errorbar(dtheta,  dtheta*dvalue, dtheta*derror, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Randoms Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyRandoms_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'randoms')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight


        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'randoms'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_randoms'), overwrite=True)

        # Also make a plot of the data

