

class TXShearCalibrationMetacal(PipelineStage):
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
    ]

    output = [
        ('calibrated_shear_catalog', ShearCatalog),
    ]

    config_options = {
        "subtract_mean_shear": True,
    }

    def run(self):
        # load global calibration parameters
        # loop through data selecting

        output = self.setup_output(nbin)

        if self.rank > nbin:
            print(f"NOTE: Processor {self.rank} will be idle as parallelization is by bin in TXShearCalibration")


        for s, e, data in it:

            # Parallelization is by tomographic here, because
            # otherwise we don't know how the 
            for i in self.split_tasks_by_rank(range(nbin)):




    def get_calibrated_catalog_bin(self, data, meta, i):
        """
        Calculate the metacal correction factor for this tomographic bin.
        """

        mask = (data['source_bin'] == i)

        if self.config['use_true_shear']:
            g1 = data[f'true_g1'][mask]
            g2 = data[f'true_g2'][mask]

        elif self.config['shear_catalog_type']=='metacal':
            # We use S=0 here because we have already included it in R_total
            g1, g2 = apply_metacal_response(data['R'][i], 0.0, data['mcal_g1'][mask], data['mcal_g2'][mask])

        elif self.config['shear_catalog_type']=='lensfit':
            #By now, by default lensfit_m=None for KiDS, so one_plus_K will be 1
            g1, g2, weight, one_plus_K = apply_lensfit_calibration(
                g1 = data['g1'][mask],
                g2 = data['g2'][mask],
                weight = data['weight'][mask],
                sigma_e = data['sigma_e'][mask], 
                m = data['m'][mask]
            )

        else:
            raise ValueError(f"Please specify metacal or lensfit for shear_catalog in config.")
            
        # Subtract mean shears, if needed.  These are calculated in source_selector,
        # and have already been calibrated, so subtract them after calibrated our sample.
        # Right now we are loading the full catalog here, so we could just take the mean
        # at this point, but in future we would like to move to just loading part of the
        # catalog.
        if self.config['subtract_mean_shear']:
            # Cross-check: print out the new mean.
            # In the weighted case these won't actually be equal
            mu1 = g1.mean()
            mu2 = g2.mean()

            # If we flip g2 we also have to flip the sign
            # of what we subtract
            g1 -= meta['mean_e1'][i]
            g2 -= meta['mean_e2'][i]

            # Compare to final means.
            nu1 = g1.mean()
            nu2 = g2.mean()
            print(f"Subtracting mean shears for bin {i}")
            print(f"Means before: {mu1}  and  {mu2}")
            print(f"Means after:  {nu1}  and  {nu2}")
            print("(In the weighted case the latter may not be exactly zero)")


        return g1, g2, mask