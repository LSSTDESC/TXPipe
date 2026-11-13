# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog
from ...utils.calibrators import Calibrator
from ...utils import DynamicSplitter
from collections import defaultdict
import yaml
import ceci
import itertools


class CLClusterBinningRedshiftRichness(PipelineStage):
    name = "CLClusterBinningRedshiftRichness"
    parallel = False
    inputs = [("cluster_catalog", HDFFile)]
    outputs = [("cluster_catalog_tomography", HDFFile)]
    config_options = {
        "zedge": [0.2, 0.4, 0.6, 0.8, 1.0],
        "richedge": [5., 10., 20.],
        "initial_size": 100_000,
        "chunk_rows": 100_000,
    }
    def run(self):        
        initial_size = self.config["initial_size"]
        chunk_rows = self.config["chunk_rows"]
        
        zedge = np.array(self.config['zedge'])
        richedge = np.array(self.config['richedge'])
        
        nz = len(zedge) - 1
        nr = len(richedge) - 1
                    
        # add infinities to either end to catch objects that spill out
        zedge = np.concatenate([[-np.inf], zedge, [np.inf]])
        richedge = np.concatenate([[-np.inf], richedge, [np.inf]])
        
        # all pairs of z bin, richness bin indices
        bins = list(itertools.product(range(nz), range(nr)))
        bin_names = {f"zbin_{i}_richbin_{j}":initial_size for i,j in bins}
        #bin_names = [f"zbin_{i}_richbin_{j}" for i,j in bins]

        
        # Columns we want to save for each object
        cols = ['cluster_id', 'dec', 'ra', 'redshift', 'redshift_err', 'richness', 'richness_err', 'scaleval']


        f = self.open_output("cluster_catalog_tomography")
        g = f.create_group("cluster_bin")
        g.attrs['nr'] = nr
        g.attrs['nz'] = nz
        splitter = DynamicSplitter(g, "bin", cols, bin_names)

        # Make an iterator that will read a chunk of data at a time
        it = self.iterate_hdf("cluster_catalog", "clusters", cols, chunk_rows)

        # Loop through the chunks of data; each time `data` will be a
        # dictionary of column names -> numpy arrays
        for _, _, data in it:
            n = len(data["redshift"])

            # Figure out which bin each halo it in, if any, starts at 0
            zbin = np.digitize(data['redshift'], zedge) - 2
            richbin = np.digitize(data["richness"], richedge) - 2

            # Find which bin each object is in, or None
            for zi in range(0, nz):
                for ri in range(0, nr):
                    w = np.where((zbin == zi) & (richbin == ri))
                    # if there are no objects in this bin in this chunk,
                    # then we skip the rest
                    if w[0].size == 0:
                        continue

                    # Otherwise we extract the bit of this chunk of
                    # data that is in this bin and have our splitter
                    # object write it out.
                    d = {name:col[w] for name, col in data.items()}
                    bin_name = f"zbin_{zi}_richbin_{ri}" #TO CHANGE ?
                    splitter.write_bin(d, bin_name)

        # Truncate arrays to correct size
        splitter.finish()

        # Save metadata
        for (i, j), name in zip(bins, bin_names):
            metadata = splitter.subgroups[name].attrs
            metadata['rich_min'] = richedge[j+1]
            metadata['rich_max'] = richedge[j+2]
            metadata['z_min'] = zedge[i+1]
            metadata['z_max'] = zedge[i+2]

        f.close()
