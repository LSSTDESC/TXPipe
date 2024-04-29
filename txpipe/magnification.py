from .base_stage import PipelineStage
from .data_types import (
    YamlFile,
    HDFFile,
    FitsFile,
    PNGFile,
)
from .utils import LensNumberDensityStats, Splitter, rename_iterated
from .binning import build_tomographic_classifier, apply_classifier
import numpy as np
import warnings


class TXSSIMagnification(PipelineStage):
    """
    class for computing the magnification coefficients using SSI outputs
    Following the methodology of https://arxiv.org/abs/2012.12825
    and https://arxiv.org/abs/2209.09782
    """

    name = "TXSSIMagnification"

    inputs = [
        ("binned_lens_catalog_nomag", HDFFile),
        ("binned_lens_catalog_mag", HDFFile),
    ]

    outputs = [
        ("magnification_coefficients", HDFFile),
        ("magnification_plot", PNGFile),
    ]

    config_options = {
        "chunk_rows": 10000,
        "applied_magnification": 1.02, #magnification applied to the "magnified" SSI catalog
        #TO DO: add a way for "applied_magnification" to be determined from the SSI inputs directly
        "n_patches":20,
        "bootstrap_error":True,
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        from scipy.stats import bootstrap

        # load the catalogs
        nomag_cat = self.open_input("binned_lens_catalog_nomag")
        mag_cat   = self.open_input("binned_lens_catalog_mag")

        # get/estimate the magnification applied to each catalog
        # could put this as optional config item in TXSSIIngest
        mu = self.config["applied_magnification"]
        deltak = (1. - 1./mu)/2.
         
        cluster = self.calc_cluster_patches(nomag_cat)

        # compute number of objects in each bin in each catalog
        # (+ number of shared objects?)
        nbins = nomag_cat['lens/'].attrs['nbin_lens']
        outfile = self.setup_output(nbins)
        for ibin in range(nbins):
            print(f'Computing magnification coefficient for bin {ibin+1}')

            # single redshift bins should be low enough number 
            # count that we can load the whole sample
            w1 = nomag_cat[f'lens/bin_{ibin}/weight'][:]
            w2 = mag_cat[f'lens/bin_{ibin}/weight'][:]

            #compute mag coeff with the whole sample
            csample = self.calc_frac_change(w1, w2)/deltak
            self.write_output(outfile, csample, ibin)

            if self.config["bootstrap_error"]:
                print(f'Computing uncertainty with bootstrap')

                #compute mag coeff with subsampling patches to get a boostrap error
                patch1 = cluster.predict(
                    np.transpose(
                        [ nomag_cat[f'lens/bin_{ibin}/ra'][:],
                          nomag_cat[f'lens/bin_{ibin}/dec'][:], ]
                        )
                    )
                patch2 = cluster.predict(
                    np.transpose(
                        [ mag_cat[f'lens/bin_{ibin}/ra'][:],
                          mag_cat[f'lens/bin_{ibin}/dec'][:], ]
                        )
                    )

                #count number of objects in each patch
                label1, counts1 = np.unique(patch1, return_counts=True)
                label2, counts2 = np.unique(patch2, return_counts=True)
                assert (label1 == np.arange(cluster.n_clusters)).all(), "empty JK patches"
                assert (label2 == np.arange(cluster.n_clusters)).all(), "empty JK patches"

                boot = bootstrap([counts1, counts2], self.calc_frac_change)

                csample_boot_mean = np.mean(boot.bootstrap_distribution)/deltak
                csample_boot_err = np.std(boot.bootstrap_distribution)/np.abs(deltak)
                self.write_output_boot(outfile, csample_boot_mean, csample_boot_err, ibin)


        self.plot_results(outfile)
        outfile.close()

    def calc_cluster_patches(self, nomag_cat ):
        """
        Split the lens sample into patches 
        We are using the full no-magnification sample to do this
        but we could also use jackknife patches 
        based on this SSI run's mask if we have it...
        """
        import sklearn.cluster

        n_patches = self.config["n_patches"]
        cluster = sklearn.cluster.KMeans(n_clusters=n_patches, n_init='auto')

        # limit number of objects too 1000 per patch so stop the training 
        # from taking too long  
        max_per_patch = 1000
        maxobj = max_per_patch*n_patches

        n_obj_lens = nomag_cat[f'lens/bin_all/ra'].shape[0]

        if n_obj_lens > maxobj:
            index = np.sort(np.random.choice(n_obj_lens, size=maxobj, replace=False))
            ra = nomag_cat[f'lens/bin_all/ra'][index]
            dec = nomag_cat[f'lens/bin_all/dec'][index]
        else:
            ra = nomag_cat[f'lens/bin_all/ra'][:]
            dec = nomag_cat[f'lens/bin_all/dec'][:]
        radec = np.transpose([ra,dec])
        cluster.fit(radec)

        return cluster

    def setup_output(self, nlens):
        import h5py 
        f = self.open_output("magnification_coefficients")
        g = f.create_group("magnification")

        g.create_dataset("csample", shape=(nlens,), )
        g.create_dataset("ctotal", shape=(nlens,), )
        g.create_dataset("alpha", shape=(nlens,), )

        if self.config["bootstrap_error"]:
            g.create_dataset("csample_boot_mean", shape=(nlens,), )
            g.create_dataset("csample_boot_err", shape=(nlens,), )
        return f

    def write_output(self, outfile, csample, ibin ):
        outfile["magnification/csample"][ibin] = csample
        outfile["magnification/ctotal"][ibin]  = csample-2.
        outfile["magnification/alpha"][ibin]   = csample/2.

    def write_output_boot(self, outfile, csample_mean, csample_err, ibin ):
        outfile["magnification/csample_boot_mean"][ibin] = csample_mean
        outfile["magnification/csample_boot_err"][ibin]  = csample_err

    def plot_results(self, outfile):
        """
        plot magnifiation coeff vs bin index
        """
        import matplotlib.pyplot as plt
        
        csample = outfile["magnification/csample"][:]
        index = np.arange(len(csample))

        fig = self.open_output("magnification_plot", wrapper=True)
        plt.plot(index, csample, 'o', label='SSI', color='b')

        if self.config["bootstrap_error"]:
            mean = outfile["magnification/csample_boot_mean"][:]
            err = outfile["magnification/csample_boot_err"][:]
            plt.errorbar(index+0.1, mean, err, fmt='.', color='r', label='SSI bootstrap')
        
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.xlabel("lens bin")
        plt.ylabel(r"$C_{\rm sample}$")
        plt.legend()
        plt.axhline(0, color='k', ls='-')
        plt.axhline(2, color='k', ls='--')
        fig.close()

    @staticmethod
    def calc_frac_change(weights_nomag, weights_mag):
        N0 = np.sum(weights_nomag)
        N1 = np.sum(weights_mag)
        return (N1-N0)/N0

    @staticmethod
    def calc_frac_change_patches(counts1, counts2):
        N0 = np.sum(counts1)
        N1 = np.sum(counts2)
        return (N1-N0)/N0
