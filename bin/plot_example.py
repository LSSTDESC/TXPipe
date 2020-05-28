import matplotlib
matplotlib.use('agg')
import pyccl
import txpipe.plotting

def example():
    versions = [
        'CosmoDC2'
    ]

    sacc_files = [
        'data/cosmodc2/outputs/twopoint_data_fourier.sacc',
        ]
    cosmo = pyccl.Cosmology.read_yaml("./data/fiducial_cosmology.yml")

    figures = txpipe.plotting.full_3x2pt_plots(sacc_files, versions,
        cosmo=cosmo)

    for name, fig in figures.items():
        fig.savefig(name + '.png')


if __name__ == '__main__':
    example()