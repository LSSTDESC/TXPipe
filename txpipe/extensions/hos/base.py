from ...base_stage import PipelineStage
from ...utils import choose_pixelization
from ...data_types import SACCFile
import numpy as np


class HOSStage(PipelineStage):
    name = "HOSStage"
    def load_mask(self):
        """
        Load the mask from the input file.

        TODO:
        - Add optional downgrading of nside

        Parameters
        ----------
        None

        Returns
        -------
        mask : array
            An array of the mask in Healpix form.
            TODO: Document mask details

        pixel_scheme : object
            The pixelization scheme object.

        """
        with self.open_input("mask", wrapper=True) as f:
            info = f.read_map_info("mask")
            mask = f.read_map("mask")

        pixel_scheme = choose_pixelization(**info)
        return mask, pixel_scheme
    
    def load_overdensity_maps(self, set_bad_pixels_to_zero=False):
        """
        Load the over-density maps from the density maps file.

        Parameters
        ----------
        set_bad_pixels_to_zero : bool
            If True, clean the maps by setting NaNs and Healpix
            Unseen values to zero. Defaults to True.

        Returns
        -------
        density_maps : list
            A list of nbin_lens over-density maps.
        nbin_lens : int
            The number of lens bins.
        """
        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            density_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]
            print(f"Loaded {nbin_lens} overdensity maps")

        if set_bad_pixels_to_zero:
            density_maps = [self.clean_map(m) for m in density_maps]

        return density_maps, nbin_lens
    
    def load_convergence_maps(self):
        """
        Load the convergence maps from the convergence_maps file.

        Parameters
        ----------
        None

        Returns
        -------
        convergence_maps : list
            A list of nbin_source + 1 convergence maps, the tomographic
            bins first then the non-tomographic map.

        nbin_source : int
            The number of source bins.
        """
        with self.open_input("convergence_maps", wrapper=True) as f:
            nbin_source = f.file["maps"].attrs["nbin_source"]
            convergence_maps = [f.read_map(f"kappa_E_{b}") for b in range(nbin_source)]
            convergence_maps.append(f.read_map("kappa_E_2D"))
            print(f"Loaded {nbin_source} convergence maps")
        return convergence_maps, nbin_source        

    def clean_map(self, m):
        """
        Convert NaNs and Healpix Unseen values in a map to zero.

        Parameters
        ----------
        m : array
            The input map to be cleaned.
        Returns
        -------
        map : array
            A copy of the cleaned map with NaNs and Healpix Unseen values set to zero.
        """
        import healpy
        m = m.copy()
        m[np.isnan(m)] = 0
        m[m == healpy.UNSEEN] = 0
        return m
    
    def prepare_output_sacc(self, nbin_lens=0, nbin_source=0):
        """Set up a SACC object to be saved with important metadata.
        
        This method:
        - creates the SACC object
        - saves the provenance information
        - saves the lens and source n(z) information, if nbin_lens or nbin_source are set
        - saves metadata information

        If nbin_lens or nbin_source are set, then the list of inputs for the class must
        include lens_photoz_stack or shear_photoz_stack, respectively.

        Parameters
        ----------
        nbin_lens : int
            The number of lens bins. Default is 0, in which case no lens n(z) information is copied

        nbin_source : int
            The number of source bins. Default is 0, in which case no source n(z) information is copied
        
        """
        import sacc
        s = sacc.Sacc()

        # Save generic provenance information, such as configuration,
        # version information, and git commit hash.
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and "\n" in value:
                values = value.split("\n")
                for i, v in enumerate(values):
                    s.metadata[f"provenance/{key}_{i}"] = v
            else:
                s.metadata[f"provenance/{key}"] = value

        # If we have any lens bins, save the number density
        # information from the input lens n(z) file into the output
        # sacc file so it can be used in predictions.
        if nbin_lens > 0:
            with self.open_input("lens_photoz_stack", wrapper=True) as f:
                for i in range(nbin_lens):
                    z, Nz = f.get_bin_n_of_z(i)
                    s.add_tracer("NZ", f"lens_{i}", z, Nz)

            # Read the density of lenses per square arcmin
            with self.open_input("tracer_metadata") as f:
                lens_density = f["tracers/lens_density"][:]

            # Copy the lens density info into the SACC metadata
            for i in range(nbin_lens):
                s.metadata[f"lens_density_{i}"] = lens_density[i]

        # Do the same for the source sample, again only if requested
        if nbin_source > 0:
            with self.open_input("shear_photoz_stack", wrapper=True) as f:
                for i in range(nbin_source):
                    z, Nz = f.get_bin_n_of_z(i)
                    s.add_tracer("NZ", f"source_{i}", z, Nz)

            # We also read the source sample effective number density
            with self.open_input("tracer_metadata") as f:
                source_density = f["tracers/source_density"][:]
                n_eff = f["tracers/n_eff"][:]

            # Copy into the SACC metadata
            for i in range(nbin_source):
                s.metadata[f"source_density_{i}"] = source_density[i]
                s.metadata[f"n_eff_{i}"] = n_eff[i]

        return s
