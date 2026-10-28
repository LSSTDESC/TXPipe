#!/usr/bin/env python
"""Create a scalar_metadetect catalog from a metadetect catalog."""

import argparse
import sys
from pathlib import Path
import h5py

VARIANTS = ["ns", "1p", "1m"]


def make_scalar_metadetect_catalog(input_filename: str | Path, output_filename: str | Path) -> None:
    """Create a scalar_metadetect catalog from a metadetect catalog.

    Copies the shear/ns, shear/1p, and shear/1m groups from the input
    metadetect catalog into a new file, and marks the result as a
    scalar_metadetect catalog by setting the "catalog_type" attribute
    on its shear group.

    Args:
        input_filename: Path to the input metadetect HDF5 catalog
        output_filename: Path to write the new scalar_metadetect HDF5 catalog

    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If the input file does not look like a metadetect catalog
    """
    input_path = Path(input_filename)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with h5py.File(input_path, "r") as infile:
        missing = [v for v in VARIANTS if f"shear/{v}" not in infile]
        if missing:
            raise ValueError(
                f"{input_path} does not look like a metadetect catalog "
                f"(missing shear/{{{','.join(missing)}}})"
            )

        with h5py.File(output_filename, "w") as outfile:
            outgroup = outfile.create_group("shear")
            for variant in VARIANTS:
                print(f"Copying shear/{variant}")
                outfile.copy(infile[f"shear/{variant}"], outgroup, name=variant)

            outgroup.attrs["catalog_type"] = "scalar_metadetect"


parser = argparse.ArgumentParser(description="Create a scalar_metadetect catalog from a metadetect catalog")
parser.add_argument("input", help="Path to the input metadetect HDF5 catalog")
parser.add_argument("output", help="Path to write the new scalar_metadetect HDF5 catalog")


def main():
    args = parser.parse_args()
    make_scalar_metadetect_catalog(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
