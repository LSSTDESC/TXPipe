import h5py


def rename_datasets(group):
    items_to_rename = []
    starting_keys = list(group.keys())  # Get a list of keys to avoid runtime errors during renaming
    for key in starting_keys:
        if isinstance(group[key], h5py.Dataset) and key.startswith('mcal_'):
            new_key = key[5:]  # Remove 'mcal_' prefix
            items_to_rename.append((key, new_key))
        # Recursively process subgroups
        if isinstance(group[key], h5py.Group):
            rename_datasets(group[key])
    
    # Rename items after iteration to avoid runtime errors
    for old_key, new_key in items_to_rename:
        if new_key in starting_keys:
            print("{old} -> {new}. [OVERWRITE]".format(old=old_key, new=new_key))
            del group[new_key]  # Remove existing dataset if it exists
        else:
            print("{old} -> {new}".format(old=old_key, new=new_key))
        group[new_key] = group[old_key]
        del group[old_key]


def main(filename):
    """Remove 'mcal_' prefix from all datasets in an HDF5 file."""
    with h5py.File(filename, 'r+') as f:
        rename_datasets(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove 'mcal_' prefix from all datasets in an HDF5 file.")
    parser.add_argument('filename', type=str, help='Path to the HDF5 file to process')
    args = parser.parse_args()
    main(args.filename)
