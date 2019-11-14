import matplotlib
matplotlib.use('agg')
import pyccl
import txpipe.plotting

def example():
    versions = [
        '2.1i',
        '2.1.1i'
    ]

    sacc_files = [
        'cosmodc2-2.1i-outputs/twopoint_data.sacc',
        'cosmodc2-2.1.1i-outputs/twopoint_data.sacc',
        ]
    cosmo = pyccl.Cosmology.read_yaml("./test/fiducial_cosmology.yml")

    xip, xim, gamma, w = txpipe.plotting.full_3x2pt_plots(sacc_files, versions)

    xip.savefig('xip.png')
    xim.savefig('xim.png')
    gamma.savefig('gamma.png')
    w.savefig('w.png')


if __name__ == '__main__':
    example()