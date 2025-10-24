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
from ceci.config import StageParameter


class TXSSIMagnification(PipelineStage):
    """
    Compute the magnification coefficients using SSI outputs

    Follows the methodology of https://arxiv.org/abs/2012.12825
    and https://arxiv.org/abs/2209.09782
    """

    name = "TXSSIMagnification"
    parallel = False

    inputs = [
        ("binned_lens_catalog_nomag", HDFFile),
        ("binned_lens_catalog_mag", HDFFile),
    ]

    outputs = [
        ("magnification_coefficients", HDFFile),
        ("magnification_plot", PNGFile),
    ]

    config_options = {
        "chunk_rows": StageParameter(int, 10000, msg="Number of rows to process in each chunk."),
        # TODO: add a way for "applied_magnification" to be determined from the SSI inputs directly
        "applied_magnification": StageParameter(float, 1.02, msg="Magnification applied to the 'magnified' SSI catalog."),
        "n_patches": StageParameter(int, 20, msg="Number of patches for bootstrap error estimation."),
        "bootstrap_error": StageParameter(bool, True, msg="Whether to compute bootstrap errors."),
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        from scipy.stats import bootstrap

        # load the catalogs
        nomag_cat = self.open_input("binned_lens_catalog_nomag")
        mag_cat   = self.open_input("binned_lens_catalog_mag")

        # get/estimate the magnification applied to the magnified catalog
        # TO DO: could put this as optional config item in TXSSIIngest
        # so that it is done automatically
        mu = self.config["applied_magnification"]
        deltak = (1. - 1./mu)/2.
        
        if self.config["bootstrap_error"]:
            # split no-mag sample up into patches using K-means clustering 
            # to be used in bootstrap errors
            cluster = self.calc_cluster_patches(nomag_cat)

        # compute fractional change in number count for both samples
        # in each lens bin and for the full "2D" sample
        nbins = nomag_cat['lens/'].attrs['nbin_lens']
        outfile = self.setup_output(nbins+1)
        for ibin in range(nbins+1):
            if ibin == nbins: #last "bin" will be the full 2d sample (all bins)
                bin_label = "all"
                print(f'Computing magnification coefficient for bin_all')
            else:
                bin_label = ibin
                print(f'Computing magnification coefficient for bin {ibin+1}')

            # single redshift bins should be low enough number 
            # count that we can load the whole sample
            w1 = nomag_cat[f'lens/bin_{bin_label}/weight'][:]
            w2 = mag_cat[f'lens/bin_{bin_label}/weight'][:]

            #compute mag coeff with the whole sample
            csample = self.calc_frac_change(w1, w2)/deltak
            self.write_output(outfile, csample, ibin)

            if self.config["bootstrap_error"]:
                print(f'Computing uncertainty with bootstrap')

                #assign each object to it's patch from the K-means clustering
                #1=unmagnified, 2=magnified
                patch1 = cluster.predict(
                    np.transpose(
                        [ nomag_cat[f'lens/bin_{bin_label}/ra'][:],
                          nomag_cat[f'lens/bin_{bin_label}/dec'][:], ]
                        )
                    )
                patch2 = cluster.predict(
                    np.transpose(
                        [ mag_cat[f'lens/bin_{bin_label}/ra'][:],
                          mag_cat[f'lens/bin_{bin_label}/dec'][:], ]
                        )
                    )

                #count the (weighted) number of objects in each patch
                label1, index_array1 = np.unique(patch1, return_inverse=True)
                weighted_counts1 = np.bincount(index_array1, weights=w1)
                label2, index_array2 = np.unique(patch2, return_inverse=True)
                weighted_counts2 = np.bincount(index_array2, weights=w2)
                assert (label1 == np.arange(cluster.n_clusters)).all(), "empty bootstrap patches"
                assert (label2 == np.arange(cluster.n_clusters)).all(), "empty bootstrap patches"

                # define a function that computes fractional number count change
                # and takes only patch IDs as input (i.e. fix the counts to this z bin)
                def calc_frac_change_patches_ibin(patch_ids):
                    return self.calc_frac_change_patches(patch_ids, weighted_counts1, weighted_counts2)

                boot = bootstrap(np.array([label1]), calc_frac_change_patches_ibin)

                csample_boot_mean = np.mean(boot.bootstrap_distribution)/deltak
                csample_boot_err = np.std(boot.bootstrap_distribution)/np.abs(deltak)
                self.write_output_boot(outfile, csample_boot_mean, csample_boot_err, ibin)


        self.plot_results(outfile)
        outfile.close()

    def calc_cluster_patches(self, nomag_cat ):
        """
        Split the lens sample into patches using K-means clustering
        We are using the full no-magnification sample to do this
        but we could also use jackknife patches 
        based on this SSI run's mask if we have it...
        """
        import sklearn.cluster

        n_patches = self.config["n_patches"]
        cluster = sklearn.cluster.KMeans(n_clusters=n_patches, n_init='auto')

        # limit number of objects to 1000 per patch to stop the training 
        # from taking too long  
        max_per_patch = 1000
        maxobj = max_per_patch*n_patches

        n_obj_lens = nomag_cat[f'lens/bin_all/ra'].shape[0]

        if n_obj_lens > maxobj:
            # Here we are reading random ra,decs directly from the HDF file
            # If the speed of this gets too slow we can look into ways of speeding things up 
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
            plt.errorbar(index+0.05, mean, err, fmt='.', color='r', label='SSI bootstrap')
        
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
    def calc_frac_change_patches(patch_ids, counts1=None, counts2=None):     
        N0 = np.sum(counts1[patch_ids])
        N1 = np.sum(counts2[patch_ids])
        return (N1-N0)/N0
