import sys
import h5py

metacal_filename = sys.argv[1]
metadet_filename = sys.argv[2]

bands = "riz"
prefixes = {"00": "", "1p":"_1p", "1m":"_1m", "2p":"_2p", "2m":"_2m"}
cols = ["flags", "T", "T_err", "g1", "g2", "s2n"] +  [f"mag_{b}" for b in bands] + [f"mag_err_{b}" for b in bands]
nonprefixed_cols = ["mcal_psf_T_mean","psf_g1", "psf_g2",  "mcal_psf_g1", "mcal_psf_g2", "weight", "redshift_true", "true_g1", "true_g2", "ra", "dec", "id"]
truth_cols = []

with h5py.File(metacal_filename, "r") as infile:
    
    with h5py.File(metadet_filename, "w") as outfile:
        outfile.copy(infile["provenance"], "/provenance", expand_external=True)

        ingroup = infile["shear"]
        outgroup = outfile.create_group("shear")
        
        for p in prefixes.keys():
            outgroup.create_group(p)

        for col in cols:
            for p1, p2 in prefixes.items():
                print(f"Copying mcal_{col}{p2} -> {p1}/{col}")
                outgroup[p1].create_dataset(col, data = ingroup[f"mcal_{col}{p2}"][:])

        for col in nonprefixed_cols:
            for p1 in prefixes.keys():
                print(f"Copying {col} -> {p1}/{col}")
                outgroup[p1].create_dataset(col, data = ingroup[f"{col}"][:])

        for col in truth_cols:
            print(f"Copying {col} -> {col}")
            outgroup.create_dataset(col, data = ingroup[f"{col}"][:])



