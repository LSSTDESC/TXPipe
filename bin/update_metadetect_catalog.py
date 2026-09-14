#!/usr/bin/env python
"""Update metadetect catalog by copying shear/00 to shear/ns."""

import argparse
import sys
from pathlib import Path
import h5py



def update_catalog(filename: str | Path) -> None:
    """Update a metadetect shear catalog to use the "ns" (no shear)
    name instead of "00".

    Args:
        filename: Path to the HDF5 file to update

    Raises:
        FileNotFoundError: If the file does not exist
        KeyError: If required datasets are not found
        OSError: If the file cannot be opened or modified
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r+") as f:
        if "shear/00" not in f:
            raise ValueError(f"{filepath} is not a metadetect file")
        if "ns" in f["shear"].keys():
            sys.stderr.write("This file has already been updated.\n")
        else:
            f["shear/ns"] = f["shear/00"]

parser = argparse.ArgumentParser(description="Update metadetect catalog by copying shear/00 to shear/ns")
parser.add_argument("filename", help="Path to the HDF5 catalog file to update")

def main():
    args = parser.parse_args()
    update_catalog(args.filename)
    return 0

if __name__ == "__main__":
    sys.exit(main())
