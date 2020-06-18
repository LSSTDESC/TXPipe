from .data_types import MapsFile, PNGFile
from .base_stage import PipelineStage


class TXMapPlots(PipelineStage):
    """
    """
    name='TXMapPlots'

    inputs = [
        ('source_maps', MapsFile),
        ('lens_maps', MapsFile),
        ('density_maps', MapsFile),
        ('mask', MapsFile),
        ('aux_maps', MapsFile),
    ]
    outputs = [
        ('depth_map', PNGFile),
        ('lens_map', PNGFile),
        ('shear_map', PNGFile),
        ('flag_map', PNGFile),
        ('psf_map', PNGFile),
        ('mask_map', PNGFile),
        ('bright_object_map', PNGFile),
    ]
    config = {}

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        self.aux_plots()
        self.source_plots()
        self.lens_plots()
        self.mask_plots()

    def aux_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("aux_maps", wrapper=True)

        nbin_source = m.file['maps'].attrs['nbin_source']
        flag_max = m.file['maps'].attrs['flag_exponent_max']

        # Depth plots
        fig = self.open_output('depth_map', wrapper=True, figsize=(5,5))
        m.plot('depth/depth', view='cart')
        fig.close()

        # Bright objects
        fig = self.open_output('bright_object_map', wrapper=True, figsize=(5,5))
        m.plot('bright_objects/count', view='cart')
        fig.close()

        # Flag count plots
        fig = self.open_output('flag_map', wrapper=True, figsize=(5*flag_max, 5))
        for i in range(flag_max):
            plt.subplot(1, flag_max, i+1)
            f = 2**i
            m.plot(f'flags/flag_{f}', view='cart')
        fig.close()

        # PSF plots
        fig = self.open_output('psf_map', wrapper=True, figsize=(5*nbin_source, 10))
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)
        for i in range(nbin_source):
            plt.sca(axes[0, i])
            m.plot(f'psf/g1_{i}', view='cart')
            plt.sca(axes[1, i])
            m.plot(f'psf/g2_{i}', view='cart')
        fig.close()

    def source_plots(self):
        import matplotlib.pyplot as plt
        m = self.open_input("source_maps", wrapper=True)

        nbin_source = m.file['maps'].attrs['nbin_source']

        fig = self.open_output('shear_map', wrapper=True, figsize=(5*nbin_source, 10))
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)

        for i in range(nbin_source):
            plt.sca(axes[0, i])
            m.plot(f'g1_{i}', view='cart')
            plt.sca(axes[1, i])
            m.plot(f'g2_{i}', view='cart')
        fig.close()



    def lens_plots(self):
        import matplotlib.pyplot as plt
        m = self.open_input("lens_maps", wrapper=True)
        rho = self.open_input("density_maps", wrapper=True)
        nbin_lens = m.file['maps'].attrs['nbin_lens']

        fig = self.open_output('lens_map', wrapper=True, figsize=(5*nbin_lens, 5))
        _, axes = plt.subplots(2, nbin_lens, squeeze=False, num=fig.file.number)

        for i in range(nbin_lens):
            plt.sca(axes[0, i])            
            m.plot(f'ngal_{i}', view='cart')
            plt.sca(axes[1, i])            
            rho.plot(f'delta_{i}', view='cart')
        fig.close()

    def mask_plots(self):
        import matplotlib.pyplot as plt
        m = self.open_input("mask", wrapper=True)

        fig = self.open_output('mask_map', wrapper=True, figsize=(5,5))
        m.plot('mask', view='cart')
        fig.close()
