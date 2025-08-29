import numpy as np

def read_training_data(spec_file, bands, spec_mag_column_format, spec_redshift_column):
        fmt = spec_mag_column_format
        zfmt = spec_redshift_column
        training_data = {}
        training_data["sz"] = spec_file[zfmt][:]
        for band in bands:
            col = spec_file[fmt.format(band=band)][:]
            training_data[band] = col
        return training_data

def build_tomographic_classifier(bands, training_data_table, bin_edges, random_seed, comm):
    # Load the training data
    # Build the SOM from the training data
    from astropy.table import Table
    from sklearn.ensemble import RandomForestClassifier

    # If we are using multiple processes then only one should do the
    # classification, to ensure that everything is consistent. In that
    # case if we are not the root process, wait for them to finish and
    # receive and return their classifier
    if (comm is not None) and (comm.rank > 0):
        classifier = comm.bcast(None)
        features = comm.bcast(None)
        return classifier, features

    # Pull out the appropriate columns and combinations of the data
    print(f"Using these bands to train the tomography selector: {bands}")

    # Generate the training data that we will use
    # We record both the name of the column and the data itself
    features = []
    training_data = []
    for b1 in bands[:]:
        # First we use the magnitudes themselves
        features.append(b1)
        training_data.append(training_data_table[b1])
        # We also use the colours as training data, even the redundant ones
        for b2 in bands[:]:
            if b1 < b2:
                features.append(f"{b1}-{b2}")
                training_data.append(training_data_table[b1] - training_data_table[b2])
    training_data = np.array(training_data).T

    print("Training data for bin classifier has shape ", training_data.shape)

    # Now put the training data into redshift bins
    # We use -1 to indicate that we are outside the desired ranges
    z = training_data_table["sz"]
    training_bin = np.repeat(-1, len(z))
    print("Using these bin edges:", bin_edges)
    for i, zmin in enumerate(bin_edges[:-1]):
        zmax = bin_edges[i + 1]
        training_bin[(z > zmin) & (z < zmax)] = i
        ntrain_bin = ((z > zmin) & (z < zmax)).sum()
        print(f"Training set: {ntrain_bin} objects in tomographic bin {i}")

    # Can be replaced with any classifier
    classifier = RandomForestClassifier(
        max_depth=10,
        max_features=None,
        n_estimators=20,
        random_state=random_seed,
    )
    classifier.fit(training_data, training_bin)

    # Sklearn fitters can be pickled, which means they can also be sent through
    # mpi4py
    if comm is not None:
        comm.bcast(classifier)
        comm.bcast(features)

    return classifier, features


def apply_classifier(classifier, features, bands, shear_catalog_type, shear_data):
    """Apply the classifier to the measured magnitudes"""

    if shear_catalog_type == "metacal":
        prefixes = ["mcal_", "mcal_", "mcal_", "mcal_", "mcal_"]
        suffixes = ["", "_1p", "_2p", "_1m", "_2m"]
    elif shear_catalog_type == "metadetect":
        prefixes = ["00/", "1p/", "2p/", "1m/", "2m/"]
        suffixes = ["", "", "", "" "", ""]
    else:
        prefixes = [""]
        suffixes = [""]

    pz_data = {}

    for prefix, suffix in zip(prefixes, suffixes):
        # Pull out the columns that we have trained this bin selection
        # model on.
        data = []
        for f in features:
            # may be a single band
            if len(f) == 1:
                col = shear_data[f"{prefix}mag_{f}{suffix}"]
            # or a colour
            else:
                b1, b2 = f.split("-")
                col = (
                    shear_data[f"{prefix}mag_{b1}{suffix}"]
                    - shear_data[f"{prefix}mag_{b2}{suffix}"]
                )
            if np.all(~np.isfinite(col)):
                # entire column is NaN.  Hopefully this will get deselected elsewhere
                col[:] = 30.0
            else:
                ok = np.isfinite(col)
                col[~ok] = col[ok].max()
            data.append(col)
        data = np.array(data).T

        # Run the random forest on this data chunk
        pz_data[f"{prefix}zbin{suffix}"] = classifier.predict(data)
        if shear_catalog_type == "metacal":
            pz_data[f"zbin{suffix}"] = pz_data[f"{prefix}zbin{suffix}"]
    return pz_data
