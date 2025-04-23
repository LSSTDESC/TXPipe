from ...base_stage import PipelineStage
from ...data_types import MapsFile, SACCFile, QPNOfZFile, HDFFile
from ...utils import choose_pixelization
import numpy as np

class TXFSB(PipelineStage):
    """
    """
    name = "TXFSB"
    inputs = [
        ("density_maps", MapsFile),
        ("mask", MapsFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("filtered_squared_bispectrum", SACCFile),
    ]

    config_options = {
        "ells_per_bin": 10,
        "nfilters": 5,
        "include_n32": False,
    }

    def run(self):
        import fsb_2fields
        import healpy

        nfilters = self.config["nfilters"]
        ells_per_bin = self.config["ells_per_bin"]
        include_n32 = self.config["include_n32"]

        # TODO: add downgrading nside to something specific
        # to FSB depending on runtimes
        # Load mask information
        with self.open_input("mask", wrapper=True) as f:
            info = f.read_map_info("mask")
            mask = f.read_map("mask")

        pixel_scheme = choose_pixelization(**info)

        # Load density map information
        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            density_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]
            print(f"Loaded {nbin_lens} overdensity maps")

        filters = fsb_2fields.get_filters(nfilters, pixel_scheme.nside)

        # TODO: Parallelize this loop
        # TODO: Add tomo bin cross-correlations
        measurements = []
        for i in range(nbin_lens):
            # Convert NaNs and Healpix Unsees values to zero
            density_map = self.tidy_map(density_maps[i])

            # Initialize the main calculator object
            calculator = fsb_2fields.FSB(density_map, mask, filters, ells_per_bin=ells_per_bin)
    
            # get ancilliary information
            ells = calculator.bb.get_effective_ells()
    
            # We get the C_ell to make sure it is the same as that
            # computed in TXTwoPointFourier
            c_ells = calculator.cls_11_binned

            # For now we assume there is no covariance between the
            # tomographic bins, but if we want to calculate it we
            # could generate it by passing the pairs of maps to
            # the calculator above and calling get_full_cov on that.
            # This would also give use the cross-FSB
            cov = calculator.get_full_cov(n32=include_n32)

            # compute fsb bi-spectrum array, shape is n_filter x n_ell            
            measurement = (
                ells,
                c_ells,
                calculator.fsb_binned,
                cov
            )

            measurements.append(measurement)

        self.save_measurements(measurements)
        

    def tidy_map(self, map):
        """
        Convert NaNs and Healpix Unsees values to zero
        """
        import healpy
        map[np.isnan(map)] = 0
        map[map == healpy.UNSEEN] = 0
        return map
    
    def save_measurements(self, measurements):
        import sacc
        s = sacc.Sacc()

        nbin_lens = len(measurements)

        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            for i in range(nbin_lens):
                z, Nz = f.get_bin_n_of_z(i)
                s.add_tracer("NZ", f"lens_{i}", z, Nz)

        with self.open_input("tracer_metadata") as f:
            lens_density = f["tracers/lens_density"][:]
        
        # The two data type codes - one is standard, one is specific
        # to this statistic.
        fsb_data_type = "galaxy_density_filteredSquareBispectrum"
        cl_data_type = sacc.standard_types.galaxy_density_cl
        cov_blocks = []

        # Add measurements from each tomo bin one by one
        for i, m in enumerate(measurements):
            ells, c_ells, fsbs, cov = m

            # The 3-point functions have three tracers, and the C_ell spectra
            # only two, but in each case we are just doing auto-correlations for now
            fsb_tracers = (f"lens_{i}", f"lens_{i}", f"lens_{i}")
            cl_tracers = (f"lens_{i}", f"lens_{i}")

            n = 0
            # Save the 3pt data points
            for j in range(self.config["nfilters"]):
                for k, ell_eff in enumerate(ells):
                    s.add_data_point(fsb_data_type, fsb_tracers, fsbs[j, k], ell=ell_eff, tomo_bin=i, filter=j, ell_index=k)
                    n += 1

            # Save the C_ell data points. These should generally be the same as what is calculated
            # in TXTwoPointFourier, provided the masks are the same.
            for k, ell_eff in enumerate(ells):
                s.add_data_point(cl_data_type, cl_tracers, c_ells[k], ell=ell_eff, tomo_bin=i, ell_index=k)
                n += 1

            print(n, cov.shape)

            # Store this covariance chunk - full one is block-diagonal.
            cov_blocks.append(cov)

        s.add_covariance(cov_blocks)

        # Save metadata - the density per square arcmin first
        for i in range(nbin_lens):
            s.metadata[f"lens_density_{i}"] = lens_density[i]

        # Save provenance information
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and "\n" in value:
                values = value.split("\n")
                for i, v in enumerate(values):
                    s.metadata[f"provenance/{key}_{i}"] = v
            else:
                s.metadata[f"provenance/{key}"] = value


        output_filename = self.get_output("filtered_squared_bispectrum")
        s.save_fits(output_filename, overwrite=True)
