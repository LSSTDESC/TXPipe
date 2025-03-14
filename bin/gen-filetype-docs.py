import h5py
import tabulate
import sys

class DescriptionGenerator:
    def __init__(self):
        self.rows = []

    def __call__(self, name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        bits = name.split('/')
        groups = bits[:-1]
        name = bits[-1]
        kind = f"{obj.ndim}D {obj.dtype}"
        self.rows.append([groups, name, kind])

    def to_table(self):
        groups = [row[0] for row in self.rows]
        ngroup_max = max(len(group) for group in groups)
        headers = ["Group"] +  [""] * (ngroup_max - 1) + ["Name", "Kind", "Meaning"]
        rows = []
        for row in self.rows:
            groups = row[0] + [""] * (ngroup_max - len(row[0]))
            name = row[1]
            kind = row[2]
            meaning = ""
            rows.append(groups + [name, kind, meaning])
        return tabulate.tabulate(rows, headers=headers, tablefmt='rst')



def describe_file(filename, outfile):
    with h5py.File(filename) as f:
        gen = DescriptionGenerator()
        f.visititems(gen)
        outfile.write(gen.to_table())

files = {
    "PhotometryCatalog":        ("data/example/inputs/photometry_catalog.hdf5", "photometry"),
    "MetacalShearCatalog":      ("data/example/inputs/shear_catalog.hdf5", "metacal"),
    "LensfitShearCatalog":      ("data/example/inputs/lensfit_shear_catalog.hdf5", "lensfit"),
    "MetadetectShearCatalog":   ("data/example/inputs/metadetect_shear_catalog.hdf5", "metadetect"),
    "StarCatalog":              ("data/example/inputs/star_catalog.hdf5", "stars"),
    "TomographyCatalog":        ("data/example/outputs_metadetect/shear_tomography_catalog.hdf5", "tomography"),
    "BinnedCatalog":            ("data/example/outputs_metadetect/binned_lens_catalog.hdf5", "binned"),
    "RandomsCatalog":           ("data/example/outputs_metadetect/random_cats.hdf5", "randoms"),
    "MapsFile":                 ("data/example/outputs_metadetect/lens_maps.hdf5", "maps"),
    "MetaData":                 ("data/example/outputs_metadetect/tracer_metadata.hdf5", "metadata"),
}

if __name__ == "__main__":
    if sys.argv[1] == "all":
        for name, (filename, outfile) in files.items():
            outpath = "docs/src/file_details/" + outfile + ".rst"
            with open(outpath, "w") as f:
                f.write(f"## {outfile.capitalize()}\n\n")
                describe_file(filename, f)
                f.write("\n\n\n")
    else:            
        describe_file(sys.argv[1], sys.stdout)
        sys.stdout.write("\n")
