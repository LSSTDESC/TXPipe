import numpy as np
import copy

W = "galaxy_density_xi"
GAMMA = "galaxy_shearDensity_xi_t"
GAMMAX = "galaxy_shearDensity_xi_x"
XIP = "galaxy_shear_xi_plus"
XIM = "galaxy_shear_xi_minus"
EE = "galaxy_shear_cl_ee"
DD = "galaxy_density_cl"
ED = "galaxy_shearDensity_cl_e"

types = {
    W: ('theta','lens','lens'),
    GAMMA: ('theta','source','lens'),
    GAMMAX: ('theta','source','lens'),
    XIP: ('theta','source','source'),
    XIM: ('theta','source','source'),
    EE: ('ell', 'source','source'),
    DD: ('ell', 'lens','lens'),
    ED: ('ell', 'source','lens'),
}


def make_axis(i, j, nx, ny, axes):
    import matplotlib.pyplot as plt
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


def apply_galaxy_bias_ggl(obs, theory, xi):
    theory = copy.deepcopy(theory)

    nbin_source = obs['nbin_source']
    nbin_lens = obs['nbin_lens']
    bias = np.zeros(nbin_lens)

    # Best fit biases as a function of scale
    for i in range(nbin_lens):
        ell_obs, cl_obs = obs[DD, i, i]
        ell_theory, cl_theory = theory[DD, i, i]

        theory_at_obs = np.interp(ell_obs, ell_theory, cl_theory)
        b2 = cl_obs / theory_at_obs
        b = np.sqrt(b2)
        mean_b = np.mean(b**0.5)
        cl_theory *= mean_b**2
        bias[i] = mean_b
        print(f"Bias {i} = {mean_b:.2f}")

    # Now apply to GGL
    for i in range(nbin_source):
        for j in range(nbin_lens):
            _, theory_cl = theory[ED, i, j]
            theory_cl *= bias[j]

    return theory





def full_3x2pt_plots(sacc_files, labels, 
                     cosmo=None, theory_sacc_files=None, theory_labels=None,
                     xi=None, fit_bias=False, figures=None, xlogscale=True):
    import sacc
    sacc_data = []
    for f in sacc_files:
        if isinstance(f, sacc.Sacc):
            sacc_data.append(f)
        else:
            sacc_data.append(sacc.Sacc.load_fits(f))
    print(sacc_data)
    obs_data = [extract_observables_plot_data(s, label) for s, label in zip(sacc_data, labels)]
    plot_theory = (cosmo is not None)


    if plot_theory:
        # By default, just plot a single theory line, not one per observable line
        # Label it "Theory"
        if theory_sacc_files is None:
            theory_sacc_data = sacc_data[:1]
            if theory_labels is None:
                theory_labels = ["Theory"]
        else:
            theory_sacc_data = [sacc.Sacc.load_fits(sacc_file) for sacc_file in theory_sacc_files]
            # But if specified, can provide multiple theory inputs, and then label them
            if theory_labels is None:
                raise ValueError("Must provide theory names if you provide theory sacc files")
        # Get the ranges from the first obs data set
        theory_data = [make_theory_plot_data(s, cosmo, obs_data[0], label, smooth=False, xi=None) 
                       for (s, label) in zip(theory_sacc_data, theory_labels)]
        if fit_bias:
            if len(theory_data) > 1:
                print("warning - fitting to just the first set of theory spectra")
            fitted_theory_data = []
            for obs, label in zip(obs_data, labels):
                theory_fit = apply_galaxy_bias_ggl(obs, theory_data[0], xi)
                theory_fit['name'] = f"Fit to {label}"
                fitted_theory_data.append(theory_fit)
            theory_data = fitted_theory_data

    else:
        theory_data = []

    if figures is None:
        figures = {}

    for t in types:
        f = figures.get(t)

    output_figures = {}
    for t in types:
        if any(obs[t] for obs in obs_data):
            f = figures.get(t)
            output_figures[t] = make_plot(t, obs_data, theory_data, fig=f, xlogscale=xlogscale)

    return output_figures
    

def axis_setup(a, i, j, ny, ymin, ymax, name):
    import matplotlib.pyplot as plt
    if j>0:
        plt.setp(a.get_yticklabels(), visible=False)
    else:
        a.set_ylabel(f"${name}$")

    if i<ny-1:
        plt.setp(a.get_xticklabels(), visible=False)

    if name.startswith(r'C_\ell'):
        a.set_xlabel(r"$C_\ell$")
    else:
        a.set_xlabel(r"$\theta$ [arcmin]")

    a.tick_params(axis='both', which='major', length=10, direction='in')
    a.tick_params(axis='both', which='minor', length=5, direction='in')

    # Fix
    a.text(0.1, 0.1, f"{j} - {i}", transform=a.transAxes)
    if i==j==0:
        a.legend()
    a.set_ylim(ymin, ymax)


def make_plot(corr, obs_data, theory_data, fig=None, xlogscale=True):
    import matplotlib.pyplot as plt
    nbin_source = obs_data[0]['nbin_source']
    nbin_lens = obs_data[0]['nbin_lens']

    ny = nbin_source if types[corr][1] == 'source' else nbin_lens
    nx = nbin_source if types[corr][2] == 'source' else nbin_lens


    if corr == XIP:
        name = r"\xi_+(\theta)"
        ymin = 5e-7
        ymax = 9e-5
        auto_only = False
        half_only = True
    elif corr == XIM:
        name = r"\xi_-(\theta)"
        ymin = 5e-7
        ymax = 9e-5
        auto_only = False
        half_only = True
    elif corr == GAMMA:
        ymin = 5e-7
        ymax = 2e-2
        name = r'\gamma_T(\theta)'
        auto_only = False
        half_only = False
    elif corr == W:
        ymin = 2e-4
        ymax = 1e-0
        name = r'w(\theta)'
        auto_only = True
        half_only = False
    elif corr == EE:
        name = r"C_\ell^{EE}"
        ymin = 2e-12
        ymax = 9e-8
        auto_only = False
        half_only = True
    elif corr == ED:
        ymin = 2e-10
        ymax = 2e-6
        name = r"C_\ell^{ED}"
        auto_only = False
        half_only = False
    elif corr == DD:
        ymin = 2e-8
        ymax = 1e-4
        name = r"C_\ell^{DD}"
        auto_only = True
        half_only = False
    elif corr == GAMMAX:
        ymin = 5e-7
        ymax = 2e-2
        name = r"\gamma_X(\theta)"
        auto_only = False
        half_only = False

    plt.rcParams['font.size'] = 14
    f = fig if fig is not None else plt.figure(figsize=(nx*3.5, ny*3))
    ax = {}
    
    axes = f.subplots(ny, nx, sharex='col', sharey='row', squeeze=True)
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
                res = obs[(corr, i, j)]
                if len(res) == 2:
                    theta, xi = res
                    if xlogscale:
                        l, = a.loglog(theta, xi, 'x', label=obs['name'])
                        a.loglog(theta, -xi, 's', color=l.get_color())
                    else:
                        l, = a.plot(theta, xi, 'x', label=obs['name'])
                        a.plot(theta, -xi, 's', color=l.get_color())
                        a.set_yscale('log')
                else:
                    theta, xi, cov = res
                    err = cov.diagonal()**0.5
                    a.errorbar(theta, xi, err, fmt='.', label=obs['name'], capsize=5)
                    a.set_yscale('log')
                    if xlogscale:
                        a.set_xscale('log')

            for theory in theory_data:
                theta, xi = theory[(corr, i, j)]
                if xlogscale:
                    a.loglog(theta, xi, '-', label=theory['name'])
                else:
                    a.plot(theta, xi, '-', label=theory['name'])

            axis_setup(a, i, j, ny, ymin, ymax, name)
            if corr in [EE, ED, DD]:
                a.set_xlim(90, 1500)

    f.suptitle(rf"TXPipe ${name}$")

    # plt.tight_layout()
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return plt.gcf()

def smooth_nz(nz):
    return np.convolve(nz, np.exp(-0.5*np.arange(-4,5)**2)/2**2, mode='same')


def extract_observables_plot_data(data, label):
    obs = {'name': label}

    nbin_source = len([t for t in data.tracers if t.startswith('source')])
    nbin_lens   = len([t for t in data.tracers if t.startswith('lens')])
    nbin_max = max(nbin_source, nbin_lens)
    has_cov = data.has_covariance()

    obs['nbin_source'] = nbin_source
    obs['nbin_lens'] = nbin_lens

    for t in types:
        # ignore any other data we don't care about
        if t not in data.get_data_types():
            obs[t] = False
            continue

        obs[t] = True
        a, b1, b2 = types[t]


        for i in range(nbin_max):
            for j in range(nbin_max):
                B1 = f"{b1}_{i}"
                B2 = f"{b2}_{j}"
                if a == 'theta':
                    res = data.get_theta_xi(t, B1, B2, return_cov=has_cov)
                else:
                    res = data.get_ell_cl(t, B1, B2, return_cov=has_cov)

                if res[0].size > 0:
                    obs[(t, i, j)] = res
    return obs

def make_theory_plot_data(data, cosmo, obs, label, smooth=True, xi=None):
    import pyccl

    theory = {'name': label}
    xi = ('galaxy_density_xi' in data.get_data_types()) if xi is None else xi

    nbin_source = obs['nbin_source']
    nbin_lens   = obs['nbin_lens']

    ell = np.unique(np.logspace(np.log10(2),5,400).astype(int))

    tracers = {}

    # Make the lensing tracers
    for i in range(nbin_source):
        name = f'source_{i}'
        Ti = data.get_tracer(name)
        nz = smooth_nz(Ti.nz) if smooth else Ti.nz

        # Convert to CCL form
        tracers[name] = pyccl.WeakLensingTracer(cosmo, (Ti.z, nz))

    # And the clustering tracers
    for i in range(nbin_lens):
        name = f'lens_{i}'
        Ti = data.get_tracer(name)
        nz = smooth_nz(Ti.nz) if smooth else Ti.nz

        # Convert to CCL form
        tracers[name] = pyccl.NumberCountsTracer(cosmo, has_rsd=False, 
            dndz=(Ti.z, nz), bias=(Ti.z, np.ones_like(Ti.z)))


    for i in range(nbin_source):
        for j in range(i+1):
            print(f"Computing theory lensing-lensing ({i},{j})")

            # compute power spectra
            cl = pyccl.angular_cl(cosmo, tracers[f'source_{i}'], tracers[f'source_{j}'], ell)
            theory[(EE, i, j)] = ell, cl

            # Optionally also compute correlation functions
            if xi:
                theta, *_ = obs[(XIP, i, j)]
                theory[(XIP, i, j)] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, corr_type='L+')
                theory[(XIM, i, j)] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, corr_type='L-')

    for i in range(nbin_lens):
        print(f"Computing theory density-density ({i},{i})")

        # compute power spectra
        cl = pyccl.angular_cl(cosmo, tracers[f'lens_{i}'], tracers[f'lens_{i}'], ell)
        theory[(DD, i, i)] = ell, cl

        # Optionally also compute correlation functions
        if xi:
            theta, *_ = obs[(W, i, i)]
            theory[W, i, i] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, corr_type='GG')


    for i in range(nbin_source):
        
        for j in range(nbin_lens):
            print(f"Computing theory lensing-density (S{i},L{j})")

            # compute power spectra
            cl = pyccl.angular_cl(cosmo, tracers[f'source_{i}'], tracers[f'lens_{j}'], ell)
            theory[(ED, i, j)] = ell, cl

            # Optionally also compute correlation functions
            if xi:
                theta, *_ = obs[(GAMMA, i, j)]
                theory[GAMMA, i, j] = theta, pyccl.correlation(cosmo, ell, cl, theta/60, corr_type='GL')

    return theory

