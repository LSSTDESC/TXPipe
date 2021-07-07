import numpy as np
from ...base_stage import PipelineStage
from ...data_types import SACCFile, PNGFile


class CMCorrelationsPlot(PipelineStage):
    name = "CMCorrelationsPlot"
    inputs = [("cluster_mag_correlations", SACCFile),]
    outputs = [
        ("cluster_mag_halo_halo_plot", PNGFile),
        ("cluster_mag_halo_bg_plot", PNGFile)

    ]
    config_options = {}
    def run(self):
        import sacc
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        S = sacc.Sacc.load_fits(self.get_input("cluster_mag_correlations"))

        nm = S.metadata['nm']
        nz = S.metadata['nz']

        self.halo_halo_plot(S, nm, nz)
        self.halo_bg_plot(S, nm, nz)



    def halo_halo_plot(self, S, nm, nz):
        import matplotlib.pyplot as plt

        f = self.open_output('cluster_mag_halo_halo_plot', wrapper=True, figsize=(nm*5,nz*5))
        fig = f.file
        axes = fig.subplots(nm, nz, sharex='col', sharey=False, squeeze=False)
        for i in range(nz):
            for j in range(nm):
                ax = axes[i, j]
                tracer = f'halo_{i}_{j}'
                theta = S.get_tag('theta', 'halo_halo_density_xi', (tracer, tracer))
                error = S.get_tag('error', 'halo_halo_density_xi', (tracer, tracer))
                mmin = S.get_tag('mass_min', 'halo_halo_density_xi', (tracer, tracer))[0] / 1e13
                mmax = S.get_tag('mass_max', 'halo_halo_density_xi', (tracer, tracer))[0] / 1e13
                zmin = S.get_tag('z_min', 'halo_halo_density_xi', (tracer, tracer))[0]
                zmax = S.get_tag('z_max', 'halo_halo_density_xi', (tracer, tracer))[0]
                if not len(theta):
                    continue
                xi = S.get_mean('halo_halo_density_xi', (tracer, tracer))

                ax.errorbar(theta, xi, error, fmt='r.')
                ax.set_xscale('log')
                ax.axhline(0, color='k')
                ax.text(0.5, 0.9, f'z = {zmin:.2f} -- {zmax:.2f}\nM = ({mmin:.2f} -- {mmax:.2f}) $\\times 10^{{13}}$', transform=ax.transAxes)

                ax.set_title(f"Halo bin {i} {j}")

                # Add axis labels for the edge plots
                if j == 0:
                    ax.set_ylabel("xi")
                if i == nm - 1:
                    ax.set_xlabel("theta")
        plt.suptitle("Halo-Halo Autocorrelations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.close()

    def halo_bg_plot(self, S, nm, nz):
        import matplotlib.pyplot as plt

        f = self.open_output('cluster_mag_halo_bg_plot', wrapper=True, figsize=(nm*5,nz*5))
        fig = f.file
        axes = fig.subplots(nm, nz, sharex='col', sharey=False, squeeze=False)
        for i in range(nz):
            for j in range(nm):
                ax = axes[i, j]
                tracer = f'halo_{i}_{j}'
                theta = S.get_tag('theta', 'halo_galaxy_density_xi', ('background', tracer))
                error = S.get_tag('error', 'halo_galaxy_density_xi', ('background', tracer))
                mmin = S.get_tag('mass_min', 'halo_halo_density_xi', (tracer, tracer))[0] / 1e13
                mmax = S.get_tag('mass_max', 'halo_halo_density_xi', (tracer, tracer))[0] / 1e13
                zmin = S.get_tag('z_min', 'halo_halo_density_xi', (tracer, tracer))[0]
                zmax = S.get_tag('z_max', 'halo_halo_density_xi', (tracer, tracer))[0]
                if not len(theta):
                    continue
                xi = S.get_mean('halo_galaxy_density_xi', ('background', tracer))

                ax.errorbar(theta, xi, error, fmt='r.')
                ax.set_xscale('log')
                ax.axhline(0, color='k')
                ax.text(0.5, 0.9, f'z = {zmin:.2f} -- {zmax:.2f}\nM = ({mmin:.2f} -- {mmax:.2f}) $\\times 10^{{13}}$', transform=ax.transAxes)

                ax.set_title(f"Halo bin {i} {j}")
                if j == 0:
                    ax.set_ylabel("xi")
                if i == nm - 1:
                    ax.set_xlabel("theta")
        plt.suptitle("Halo-Background Correlations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.close()

