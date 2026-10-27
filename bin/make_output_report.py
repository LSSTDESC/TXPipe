import os
import argparse
import h5py
import tabulate
import sacc
import yaml

names = {
    "T_frac_psf_residual_hist": "Histogram of fractional error in PSF size",
    "bright_object_map_plot": "Map of distribution of bright object counts",
    "brighter_fatter_plot": "Brighter-fatter effect diagnostic plots",
    "convergence_map": "Map plot of the reconstructed convergence",
    "density_cl": "C_ell power spectrum of the density field",
    "density_xi": "Correlation function of density field",
    "density_xi_ratio": "Ratio of C_ell density correlation to fiducial theory prediction",
    "depth_map_plot": "Map of estimated n (default 5) sigma depth in a selected band",
    "e1_psf_residual_hist": "Histogram of PSF residuals in e1",
    "e2_psf_residual_hist": "Histogram of PSF residuals in e2",
    "flag_map_plot": "Map of counts of flagged objects",
    "g1_hist": "Histogram of shear g1",
    "g2_hist": "Histogram of shear g2",
    "g_T": "Trend of shear as a function of quadratic size T",
    "g_colormag": "Trend of shear as a function of colors and magnitudes ",
    "g_psf_T": "Trend of shear as a function of psf size",
    "g_psf_g": "Trend of shear as a function of psf ellipticity ",
    "g_snr": "Trend of shear as a function of galaxy signal-to-noise",
    "gammat_bright_stars_plot": "Tangential shear around bright stars",
    "gammat_dim_stars_plot": "Tangential shear around dim stars",
    "gammat_field_center_plot": "Tangential shear around field centers",
    "gammat_randoms_plot": "Tangential shear around random positions",
    "jk": "Jack-knife regions used in shot noise estimation and 2pt measurement",
    "lens_mag_hist": "Histogram of lens magnitudes",
    "lens_map_plot": "Map of lens sample counts",
    "lens_nz": "Lens sample ensemble photometric redshift n(z)",
    "lens_photoz_realizations_plot": "Realizations of lens sample photo-z",
    "lens_snr_hist": "Histogram of lens sample signal-to-noise",
    "mask_map_plot": "Map image of mask values",
    "nz_lens": "Lens sample ensemble photometric redshift n(z)",
    "nz_source": "Source sample ensemble photometric redshift n(z)",
    "psf_map_plot": "Map of the mean PSF ellipticity",
    "response_hist": "Histogram of the metadetection response values",
    "rowe0": "First Rowe statistic zero measurement plot",
    "rowe134": "Second set of Rowe stat measurement plots",
    "rowe25": "Third set of Rowe stat measurements plots",
    "shearDensity_cl": "Galaxy Galaxy-Lensing measured spectrum plot",
    "shearDensity_xi": "Galaxy Galaxy-Lensing measured correlation plot",
    "shearDensity_xi_ratio": "Galaxy Galaxy-Lensing measured correlation ratio to theory plot",
    "shearDensity_xi_x": "Galaxy Galaxy-Lensing cross-shear diagnostic plot",
    "shear_cl_ee": "Measured power spectrum E-mode plot",
    "shear_cl_ee_ratio": "Measured power spectrum E-mode ratio to theory plot",
    "shear_map_plot": "g1 and g2 maps",
    "shear_xi_minus": "Shear correlation function plot",
    "shear_xi_minus_ratio": "Shear correlation function ratio plot",
    "shear_xi_plus": "Shear correlation function plot",
    "shear_xi_plus_ratio": "Shear correlation function ratio plot",
    "source_mag_hist": "Histogram of magnitudes of source sample",
    "source_nz": "Plot of source sample ensemble photometric redshift n(z)",
    "source_photoz_realizations_plot": "Plot of realizations of the source sample ensemble photometric redshift n",
    "source_snr_hist": "Histogram of source sample signal-to-noise",
    "star_density_test": "Galaxy-star density correlation function plot",
    "star_shear_test": "Galaxy-star shear correlation function plot",
    "star_star_test": "Star-star shear correlation function plot",
    "tau0": "First PSF Tau statistic plot",
    "tau2": "Second PSF Tau statistic plot",
    "tau5": "Third PSF Tau statistic plot",
    "aperture_mass_data": "Mass aperture statistic measurements",
    "gammat_bright_stars": "Diagnostic measurements of the tangential shear around bright stars",
    "gammat_dim_stars": "Diagnostic measurements of the tangential shear around bright stars",
    "gammat_field_center": "Diagnostic measurements of the tangential shear around field centers",
    "gammat_randoms": "Diagnostic measurements of the tangential shear around random positions",
    "summary_statistics_fourier": "3x2pt C_ell spectrum measurements with covariance, suitable for parameter estimation (may be blinded)",
    "summary_statistics_real": "3x2pt correlation function measurements with covariance, suitable for parameter estimation (may be blinded)",
    "twopoint_data_fourier": "3x2pt C_ell spectrum measurements with shot noise only  (may be blinded)",
    "twopoint_data_real": "3x2pt correlation function measurements with shot noise only  (may be blinded)",
    "twopoint_data_real_raw": "Unblind 3x2pt correlation function measurements",
    "twopoint_gamma_x": "Diagnostic measurement of cross-shear around lens sample",
    "twopoint_theory_fourier": "Theory prediction for the 3x2pt C_ell spectrum based on fiducial cosmology",
    "twopoint_theory_real": "Theory prediction for the 3x2pt correlation function based on fiducial cosmology",


}

def summarize_sacc(filename):
    try:
        s = sacc.Sacc.load_fits(filename)
    except IOError:
        return ""
    lines = ["Data points: " + str(len(s))]
    lines.append(f"\n\nTracers:\n")
    for name in s.tracers:
        lines.append("- " + name)
    lines.append("\n")
    lines.append("Data:\n")
    for dt in s.get_data_types():
        ncombo = len(s.get_tracer_combinations(dt))
        ndata = s.get_mean(dt).size
        lines.append(f"* {dt}: {ncombo} tracer combinations, {ndata} data points")


    return "\n".join(lines)


def summarize_hdf5(filename):
    table_rows = []
    def summary(name, obj):
        if isinstance(obj, h5py.File):
            return
        bits = name.split("/")
        row = []
        if (bits[0] == "provenance"):
            return
        indent = name.count("/")
        name = bits[-1]

        row = ["" for _ in range(indent)]
        if isinstance(obj, h5py.Group):
            row.append(f"[{name}]")
        elif isinstance(obj, h5py.Dataset):
            row.append(name)
            row.append(str(obj.shape))
            row.append(str(obj.dtype))
        table_rows.append(row)

    with h5py.File(filename, "r") as f:
        f.visititems(summary)

    if len(table_rows) > 40:
        table_rows = table_rows[:40] + [["..."]]

    if table_rows:
        n = max([len(row) for row in table_rows])
        for row in table_rows:
            row.extend(["" for _ in range(n - len(row))])
        header = ["Group"] + [""] * (n - 4) + ["Dataset", "Shape", "Type"]
    else:
        header = []
    

    return tabulate.tabulate(table_rows, headers=header, tablefmt="github")
    
def summarize_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = ["    " + line.rstrip() for line in lines]
    return "\n".join(lines)

def summarize_png(filename):
    return fr"![{filename}]({filename})"

def summarize_yml(filename):
    with open(filename) as f:
        content = yaml.safe_load(f)
    content.pop("provenance", None)
    output = []
    for key,vals in content.items():
        if isinstance(vals, list):
            vals = [[v] for v in vals]
            rep = tabulate.tabulate(vals, headers=[key], tablefmt="github")
        else:
            rep = f"{key} = {vals}"
        rep = "\n" + rep
        output.append(rep)
    return "\n".join(output)



def main(input_directory, output_file):
    other_files = []
    with open(output_file, "w") as out:

        if not os.path.exists(input_directory) or not os.listdir(input_directory):
            out.write("# No report generated\n\n")
            out.write("No input directory, or directory is empty.")

        for filename in sorted(os.listdir(input_directory)):
            if filename.startswith("inprogress"):
                continue
            path = os.path.join(input_directory, filename)
            base, ext = os.path.splitext(path)
            key = os.path.split(base)[-1]
            if ext == ".png":
                text = summarize_png(path)
            elif ext == ".hdf5":
                text = summarize_hdf5(path) 
            elif ext == ".txt":
                text = summarize_txt(path)
            elif ext == ".sacc":
                text = summarize_sacc(path)
            elif ext == ".yml":
                text = summarize_yml(path)
            else:
                other_files.append(filename)
                continue
            if key in names:
                out.write(f"# {names[key]} ({filename})\n\n")
            else:
                out.write(f"# {filename}\n\n")
            out.write(text + "\n\n\n")
        if other_files:
            out.write("# Other files\n\n")
            out.write("- " + ("\n- ".join(other_files)))



parser = argparse.ArgumentParser(description="Summarize a directory of files")
parser.add_argument("input_directory", help="Directory to summarize")
parser.add_argument("output_file", help="Output file")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input_directory, args.output_file)
