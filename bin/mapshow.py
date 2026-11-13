
import sys
sys.path.insert(0, ".")
import txpipe
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import argparse


def get_map_bounds(m):
    """Calculate the minimum and maximum lat/lon bounds for a healpix map."""
    # Get the map resolution
    nside = hp.npix2nside(len(m))
    
    # Find pixels with valid data (not masked/nan)
    valid_pixels = np.where(np.isfinite(m) & (m != hp.UNSEEN))[0]
    
    if len(valid_pixels) == 0:
        # If no valid pixels, use full sky
        return (-90, 90), (-180, 180)
    
    # Convert pixel indices to theta, phi coordinates
    theta, phi = hp.pix2ang(nside, valid_pixels)
    
    # Convert to lat/lon in degrees
    lat = 90 - np.degrees(theta)  # theta=0 is north pole, theta=pi is south pole
    lon = np.degrees(phi)
    lon = np.where(lon > 180, lon - 360, lon)  # Convert to [-180, 180] range
    
    # Add small margin to bounds
    lat_margin = (lat.max() - lat.min()) * 0.05
    lon_margin = (lon.max() - lon.min()) * 0.05
    
    lat_range = [max(-90, lat.min() - lat_margin), min(90, lat.max() + lat_margin)]
    lon_range = [max(-180, lon.min() - lon_margin), min(180, lon.max() + lon_margin)]
    
    return lat_range, lon_range


def main():
    parser = argparse.ArgumentParser(description='Display healpix maps from TXPipe data files')
    parser.add_argument('filename', help='Path to the maps file')
    parser.add_argument('mapname', help='Name of the map to display')
    parser.add_argument('--projection', choices=['mollview', 'cartview'], default='mollview',
                        help='Projection type for the map display (default: mollview)')
    parser.add_argument('--title', help='Custom title for the plot (default: uses mapname)')
    
    args = parser.parse_args()

    # Use custom title if provided, otherwise use mapname
    title = args.title if args.title else args.mapname

    with txpipe.data_types.MapsFile(args.filename, "r") as f:
        m = f.read_healpix(args.mapname)

    if args.projection == 'mollview':
        hp.mollview(m, title=title)
    elif args.projection == 'cartview':
        lat_range, lon_range = get_map_bounds(m)
        print(f"Using lat range: {lat_range}, lon range: {lon_range}")
        hp.cartview(m, title=title, 
                   latra=lat_range, lonra=lon_range)

    plt.show()


if __name__ == "__main__":
    main()