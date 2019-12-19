
def metacal_variants(*names):
    return [
        name + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for name in names
    ]
def metacal_band_variants(bands, *names):
    return [
        name + "_" + band + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for band in bands
        for name in names
    ]

def calculate_selection_bias(g1, g2, sel_1p, sel_2p, sel_1m, sel_2m, delta_gamma):
    import numpy as np
    
    S = np.ones((2,2))
    S_11 = (g1[sel_1p].mean() - g1[sel_1m].mean()) / delta_gamma
    S_12 = (g1[sel_2p].mean() - g1[sel_2m].mean()) / delta_gamma
    S_21 = (g2[sel_1p].mean() - g2[sel_1m].mean()) / delta_gamma
    S_22 = (g2[sel_2p].mean() - g2[sel_2m].mean()) / delta_gamma
    
    # Also save the selection biases as a matrix.
    S[0,0] = S_11
    S[0,1] = S_12
    S[1,0] = S_21
    S[1,1] = S_22
    
    return S

def calculate_multiplicative_bias(g1_1p,g1_2p,g1_1m,g1_2m,g2_1p,g2_2p,g2_1m,g2_2m,sel_00,delta_gamma):
    import numpy as np 
    
    n = len(g1_1p[sel_00])
    R =  R = np.zeros((n,2,2))
    R_11 = (g1_1p[sel_00] - g1_1m[sel_00]) / delta_gamma
    R_12 = (g1_2p[sel_00] - g1_2m[sel_00]) / delta_gamma
    R_21 = (g2_1p[sel_00] - g2_1m[sel_00]) / delta_gamma
    R_22 = (g2_2p[sel_00] - g2_2m[sel_00]) / delta_gamma
    
    R[:,0,0] = R_11
    R[:,0,1] = R_12
    R[:,1,0] = R_21
    R[:,1,1] = R_22
    
    R = np.mean(R, axis=0)
    return R

def apply_metacal_response(R, S, g1, g2):
    from numpy.linalg import pinv
    import numpy as np
    
    mcal_g = np.stack([g1,g2], axis=1)
    
    R_total = R+S
    
    # Invert the responsivity matrix
    Rinv = pinv(R)
    
    mcal_g = np.dot(Rinv, np.array(mcal_g).T).T
    
    return mcal_g[:,0], mcal_g[:,1]