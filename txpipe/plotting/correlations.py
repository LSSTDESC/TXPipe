import numpy as np
import sacc
import matplotlib.pyplot as plt

W = "w"
GAMMA = "gammat"
XIP = "xip"
XIM = "xim"


def make_axis(i, j, nx, ny, axes):
    if i==0 and j==0:
        shares = {}
    elif j==0:
        shares = {'sharex': axes[0,0]}
    elif i==j:
        shares = {'sharey': axes[i,0]}
    else:
        shares = {'sharey': axes[i,0], 'sharey': axes[j,j]}

    a = plt.subplot(ny, nx, i*ny+j+1, **shares)
    axes[i,j] = a
    return a


def full_3x2pt_plots(sacc_files, labels, cosmo=None, theory_sacc_files=None, theory_labels=None):
    sacc_data = [sacc.Sacc.load_fits(sacc_file) for sacc_file in sacc_files]
    obs_data = [extract_observables_plot_data(s, label) for s, label in zip(sacc_data, labels)]

    plot_theory = (cosmo is not None)

    if plot_theory:
        # By default, just plot a single theory line, not one per observable line
        # Label it "Theory"
        if theory_sacc_files is None:
            theory_sacc_data = sacc_data[:1]
            theory_labels = ["Theory"]
        else:
            # But if specified, can provide multiple theory inputs, and then 
            if theory_labels is None:
                raise ValueError("Must provide theory names if you provide theory sacc files")
        # Get the ranges from the first obs data set
        theory_data = [make_theory_plot_data(s, cosmo, obs_data[0], label) 
                       for (s, label) in zip(theory_sacc_data, theory_labels)]
    else:
        theory_data = []


    return [make_plot(c, obs_data, theory_data)
        for c in [XIP, XIM, GAMMA, W]]
    

def axis_setup(a, i, j, ny, ymin, ymax, name):
    if j>0:
        plt.setp(a.get_yticklabels(), visible=False)
    else:
        a.set_ylabel(f"${name}$")
    if i<ny:
        plt.setp(a.get_xticklabels(), visible=False)
    else:
        plt.xlabel(r"$\theta / $ arcmin")

    a.tick_params(axis='both', which='major', length=10, direction='in')
    a.tick_params(axis='both', which='minor', length=5, direction='in')

    # Fix
    a.text(0.1, 0.1, f"Bin {i}-{j}", transform=a.transAxes)
    if i==j==0:
        a.legend()
    a.set_ylim(ymin, ymax)

def make_plot(corr, obs_data, theory_data):
    nbin_source = obs_data[0]['nbin_source']
    nbin_lens = obs_data[0]['nbin_lens']

    if corr == XIP:
        ny = nbin_source
        nx = nbin_source
        name = r"\xi_+(\theta)"
        ymin = 5e-7
        ymax = 9e-5
        auto_only = False
        half_only = True
    elif corr == XIM:
        ny = nbin_source
        nx = nbin_source
        name = r"\xi_-(\theta)"
        ymin = 5e-7
        ymax = 9e-5
        auto_only = False
        half_only = True
    elif corr == GAMMA:
        ny = nbin_source
        nx = nbin_lens
        ymin = 5e-7
        ymax = 2e-3
        name = r'\gamma_T(\theta)'
        auto_only = False
        half_only = False
    elif corr == W:
        ny = nbin_lens
        nx = nbin_lens
        ymin = 2e-4
        ymax = 1e-1
        name = r'\gamma_T(\theta)'
        auto_only = True
        half_only = False

    plt.rcParams['font.size'] = 14
    f = plt.figure(figsize=(nx*3, ny*3))
    ax = {}
    
    axes = f.subplots(ny, nx, sharex='col', sharey='row', squeeze=False)
    for i in range(ny):
        if auto_only:
            J = [i]
        elif half_only:
            J = range(i+1)
        else:
            J = range(nx)
        for j in range(nx):
            a = axes[i,j]
            if j not in J:
                f.delaxes(a)
                continue

            for obs in obs_data:
                theta, xi = obs[(corr, i, j)]
                a.loglog(theta, xi, '+', label=obs['name'])

            for theory in theory_data:
                theta, xi = theory[(corr, i, j)]
                a.loglog(theta, xi, '-', label=theory['name'])

            axis_setup(a, i, j, ny, ymin, ymax, name)

    f.suptitle(rf"TXPipe ${name}$")

    # plt.tight_layout()
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    return plt.gcf()

def smooth_nz(nz):
    return np.convolve(nz, np.exp(-0.5*np.arange(-4,5)**2)/2**2, mode='same')


def extract_observables_plot_data(data, label):
    obs = {'name': label}

    nbin_source = len([t for t in data.tracers if t.startswith('source')])
    nbin_lens   = len([t for t in data.tracers if t.startswith('lens')])

    obs['nbin_source'] = nbin_source
    obs['nbin_lens'] = nbin_lens

    for i in range(nbin_source):
        for j in range(i+1):
            obs[(XIP, i, j)] = data.get_theta_xi('galaxy_shear_xi_plus', f'source_{i}', f'source_{j}')
            obs[(XIM, i, j)] = data.get_theta_xi('galaxy_shear_xi_minus', f'source_{i}', f'source_{j}')

    for i in range(nbin_lens):
        obs[W, i, i] = data.get_theta_xi('galaxy_density_xi', f'lens_{i}', f'lens_{i}')


    for i in range(nbin_source):
        for j in range(nbin_lens):
            obs[(GAMMA, i, j)] = data.get_theta_xi('galaxy_shearDensity_xi_t', f'source_{i}', f'lens_{j}')

    return obs

def make_theory_plot_data(data, cosmo, obs, label, smooth=True):
    import pyccl

    theory = {'name': label}

    nbin_source = obs['nbin_source']
    nbin_lens   = obs['nbin_lens']

    ell = np.unique(np.logspace(np.log10(2),5,400).astype(int))

    tracers = {}
    for i in range(nbin_source):
        name = f'source_{i}'
        Ti = data.get_tracer(name)
        if smooth:
            nz = smooth_nz(Ti.nz)
        else:
            nz = Ti.nz
        tracers[name] = pyccl.WeakLensingTracer(cosmo, (Ti.z, nz))

    for i in range(nbin_lens):
        name = f'lens_{i}'
        Ti = data.get_tracer(name)
        if smooth:
            nz = smooth_nz(Ti.nz)
        else:
            nz = Ti.nz
        tracers[name] = pyccl.NumberCountsTracer(cosmo, has_rsd=False, 
            dndz=(Ti.z, nz), bias=(Ti.z, np.ones_like(Ti.z)))


    for i in range(nbin_source):
        for j in range(i+1):
            theta, _ = obs[(XIP, i, j)]
            cl = pyccl.angular_cl(cosmo, tracers[f'source_{i}'], tracers[f'source_{j}'], ell)
            print(f"Computing theory xip/m ({i},{j})")
            theory[(XIP, i, j)] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, 'L+')
            theory[(XIM, i, j)] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, 'L-')

    for i in range(nbin_lens):
        theta, _ = obs[(W, i, i)]
        print(f"Computing theory w ({i},{i})")
        cl = pyccl.angular_cl(cosmo, tracers[f'source_{i}'], tracers[f'source_{j}'], ell)
        theory[W, i, i] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, 'GG')


    for i in range(nbin_source):
        for j in range(nbin_lens):
            theta, _ = obs[(GAMMA, i, j)]
            print(f"Computing theory gamma_t ({i},{j})")
            cl = pyccl.angular_cl(cosmo, tracers[f'source_{i}'], tracers[f'lens_{j}'], ell)
            theory[GAMMA, i, j] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, 'GL')

    return theory

