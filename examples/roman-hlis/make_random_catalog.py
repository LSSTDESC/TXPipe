#!/usr/bin/env python
"""Generate a random catalog matched to the lens catalog footprint.

Builds a HEALPix mask from the occupied pixels in the lens catalog,
then scatters random points uniformly within those pixels.
Output format matches binned_random_catalog.hdf5.

Usage:
    python make_random_catalog.py [--lens LENS] [--output OUTPUT]
                                  [--nside NSIDE] [--factor FACTOR]
"""

import argparse
import numpy as np
import h5py
import healpy as hp

output_dir = '/pscratch/sd/c/chihway/TXPipe/data/example/outputs_roman_hlis/'
DEFAULTS = dict(
    lens    = output_dir+"binned_lens_catalog.hdf5",
    output = output_dir+"e2e/binned_random_catalog_new.hdf5",
    nside  = 512,   # HEALPix resolution for footprint mask
    factor = 20,    # randoms per lens object
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens",   default=DEFAULTS["lens"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--nside",  type=int,   default=DEFAULTS["nside"],
                   help="HEALPix nside for footprint mask (default: 512)")
    p.add_argument("--factor", type=float, default=DEFAULTS["factor"],
                   help="Number of randoms per lens object (default: 20)")
    return p.parse_args()


def count_bins(f):
    return len([k for k in f["lens"].keys()
                if k.startswith("bin_") and k != "bin_all"])


def occupied_pixels(ra, dec, nside):
    """Return unique HEALPix pixels (RING) occupied by the given positions."""
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    return np.unique(pix)


def randoms_in_pixels(pixels, nside, n):
    """Generate n random (ra, dec) points uniformly inside the given pixels.

    Uses rejection sampling: draw uniform points on the sphere within the
    bounding box of the footprint and keep those that land in an occupied pixel.
    """
    pixel_set = set(pixels)

    # bounding box in ra/dec
    cen_ra, cen_dec = hp.pix2ang(nside, pixels, lonlat=True)
    pad = np.degrees(hp.nside2resol(nside))   # ~1 pixel of padding
    ra_min,  ra_max  = cen_ra.min()  - pad, cen_ra.max()  + pad
    dec_min, dec_dec = cen_dec.min() - pad, cen_dec.max() + pad

    # uniform sampling on the sphere within the dec band:
    # draw z = sin(dec) uniformly to get uniform solid angle
    z_min = np.sin(np.radians(dec_min))
    z_max = np.sin(np.radians(dec_dec))

    ra_out  = np.empty(n, dtype=np.float32)
    dec_out = np.empty(n, dtype=np.float32)
    filled  = 0

    while filled < n:
        need    = (n - filled) * 2        # oversample to reduce iterations
        ra_try  = np.random.uniform(ra_min, ra_max, need) % 360.0
        dec_try = np.degrees(np.arcsin(np.random.uniform(z_min, z_max, need)))
        pix_try = hp.ang2pix(nside, ra_try, dec_try, lonlat=True)
        keep    = np.array([p in pixel_set for p in pix_try])
        n_keep  = keep.sum()
        take    = min(n_keep, n - filled)
        ra_out [filled:filled + take] = ra_try [keep][:take]
        dec_out[filled:filled + take] = dec_try[keep][:take]
        filled += take

    return ra_out, dec_out


def main():
    args = parse_args()
    rng_seed = 42
    np.random.seed(rng_seed)

    with h5py.File(args.lens, "r") as f:
        n_bins = count_bins(f)
        print(n_bins)

        # build a single union mask from all bins so every bin's randoms
        # cover the full footprint (same as the pipeline convention)
        all_ra  = np.concatenate([f[f"lens/bin_{b}/ra"][:]  for b in range(n_bins)])
        all_dec = np.concatenate([f[f"lens/bin_{b}/dec"][:] for b in range(n_bins)])

        bin_sizes = [len(f[f"lens/bin_{b}/ra"]) for b in range(n_bins)]

    print(f"Building footprint mask at nside={args.nside} "
          f"from {len(all_ra):,} lens objects...")
    union_pixels = occupied_pixels(all_ra, all_dec, args.nside)
    print(f"  {len(union_pixels):,} occupied pixels  "
          f"({hp.nside2pixarea(args.nside, degrees=True)*len(union_pixels):.2f} deg²)")

    with h5py.File(args.output, "w") as out:
        grp = out.create_group("randoms")
        grp.attrs["nbin"] = n_bins

        for b in range(n_bins):
            n_rand = int(bin_sizes[b] * args.factor)
            print(f"  bin {b}: {bin_sizes[b]:,} lenses → {n_rand:,} randoms", flush=True)

            ra, dec = randoms_in_pixels(union_pixels, args.nside, n_rand)

            sg = grp.create_group(f"bin_{b}")
            sg.create_dataset("ra",  data=ra)
            sg.create_dataset("dec", data=dec)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
