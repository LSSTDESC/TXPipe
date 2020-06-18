import numpy as np
import yaml
from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, MapsFile, HDFFile, YamlFile


class TXTracerMetadata(PipelineStage):
    """
    This stage doesn't actually calculate anything, it just
    collates together metadata about our sources, so that we
    don't need to pass around catalog sized objects as much.
    """
    name = "TXTracerMetadata"

    inputs = [
        ('shear_tomography_catalog', TomographyCatalog),
        ('lens_tomography_catalog', TomographyCatalog),
        ('mask', MapsFile),
    ]

    outputs = [
        ('tracer_metadata', HDFFile),
        ('tracer_metadata_yml', YamlFile), #human-readable version
    ]

    def run(self):
        # Read the area
        area = self.read_area()

        area_sq_arcmin = area * 60**2
        shear_tomo_file = self.open_input('shear_tomography_catalog')
        lens_tomo_file = self.open_input('lens_tomography_catalog')

        meta_file = self.open_output('tracer_metadata')

        def copy(tomo, in_section, out_section, name):
            x = tomo[f'{in_section}/{name}'][:]
            meta_file.create_dataset(f'{out_section}/{name}', data=x)

        def copy_attrs(tomo, name, out_name):
            for k,v in tomo[name].attrs.items():
                meta_file[out_name].attrs[k] = v


        copy(shear_tomo_file, 'metacal_response', 'tracers', 'R_gamma_mean')
        copy(shear_tomo_file, 'metacal_response', 'tracers', 'R_S')
        copy(shear_tomo_file, 'metacal_response', 'tracers', 'R_total')
        copy(shear_tomo_file, 'tomography', 'tracers', 'N_eff')
        copy(lens_tomo_file, 'tomography', 'tracers', 'lens_counts')
        copy(shear_tomo_file, 'tomography', 'tracers', 'sigma_e')
        copy(shear_tomo_file, 'tomography', 'tracers', 'source_counts')

        N_eff = shear_tomo_file['tomography/N_eff'][:]
        n_eff = N_eff / area_sq_arcmin

        lens_counts = lens_tomo_file['tomography/lens_counts'][:]
        source_counts = shear_tomo_file['tomography/source_counts'][:]

        lens_density = lens_counts / area_sq_arcmin
        source_density = source_counts / area_sq_arcmin

        meta_file.create_dataset('tracers/n_eff', data=n_eff)
        meta_file.create_dataset('tracers/lens_density', data=lens_density)
        meta_file.create_dataset('tracers/source_density', data=source_density)
        meta_file['tracers'].attrs['area'] = area
        meta_file['tracers'].attrs['area_unit'] = 'deg^2'
        meta_file['tracers'].attrs['density_unit'] = 'arcmin^{-2}'
        copy_attrs(shear_tomo_file,'tomography', 'tracers')
        copy_attrs(lens_tomo_file,'tomography', 'tracers')

        meta_file.close()

        # human readable version
        yaml_out_name = self.get_output('tracer_metadata_yml')
        metadata = {
            'lens_density': lens_density.tolist(),
            'source_density': source_density.tolist(),
            'sigma_e': shear_tomo_file['tomography/sigma_e'][:].tolist(),
            'n_eff': n_eff.tolist(),
            'R_gamma_mean': shear_tomo_file['metacal_response/R_gamma_mean'][:].tolist(),
            'R_S': shear_tomo_file['metacal_response/R_S'][:].tolist(),
            'R_total': shear_tomo_file['metacal_response/R_total'][:].tolist(),
            'lens_counts': lens_counts.tolist(),
            'source_counts': source_counts.tolist(),
            'area': float(area),
            'area_unit': 'deg^2',
            'density_unit': 'arcmin^{-2}',
        }
        f = open(yaml_out_name, 'w')
        yaml.dump(metadata, f)
        f.close()

        shear_tomo_file.close()
        lens_tomo_file.close()


    def read_area(self):
        with self.open_input('mask', wrapper=True) as f:
            m = f.read_map('mask')
        print(m.max())
        f_sky = (m > 0).sum() / m.size
        area_sq_deg = f_sky * 41252.96125
        print(area_sq_deg)
        return area_sq_deg
