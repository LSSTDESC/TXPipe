def sample_coords(hsp_map, clusters, size):
    rz = clusters[['richness', 'redshift', 'scaleval']].sample(
        n=size, replace=True, random_state=43
    ).reset_index(drop=True)

    ra_out = np.empty(size)
    dec_out = np.empty(size)
    filled = 0

    while filled < size:
        needed = size - filled
        batch = max(needed, 1000)
        
        ra, dec = hs.make_uniform_randoms(hsp_map, batch)
        pix = hp.ang2pix(4096, ra, dec, lonlat=True) # hard coding nside for now
        
        # redshift masking
        # z_at_pos = hsp_map[pix]
        z_at_pos = 50 # skipping the redshift masking

        # We compare against the redshifts for the *slots we still need to fill*
        z_needed = rz['redshift'].values[filled:filled + batch]
        # Guard against batch > remaining slots
        n_compare = min(batch, needed)
        mask = rz['redshift'].values[filled:filled + n_compare] <= z_at_pos#[:n_compare]

        n_accept = mask.sum()
        if n_accept == 0:
            continue

        take = min(n_accept, needed)
        ra_out[filled:filled + take] = ra[:n_compare][mask][:take]
        dec_out[filled:filled + take] = dec[:n_compare][mask][:take]

        # Compact accepted rz rows into their final positions
        accepted_idx = np.where(mask)[0][:take]
        for j, idx in enumerate(accepted_idx):
            if filled + j < size:
                rz.iloc[filled + j] = rz.iloc[filled + idx]

        filled += take

    coords = pd.DataFrame({'ra': ra_out, 'dec': dec_out})
    rz['weight']= rz['richness'] / rz['scaleval']
    return pd.concat([coords, rz], axis=1)


def create_random_cat(cluster_cat, hsp_map, cutoff=20):
    size = len(cluster_cat) * 100
    random_df = sample_coords(hsp_map, cluster_cat, size)
    # From this point we use the scaleval and richness to get an effective richness,
    # then compare this to a cutoff richness (a la https://arxiv.org/abs/1601.00621) that acts like a loose detection mask,
    # then we only keep the randoms that pass this masking
    # Still needs a geometric mask on top of this 
    random_df = random_df.loc[random_df['weight'] > cutoff]
    return random_df
