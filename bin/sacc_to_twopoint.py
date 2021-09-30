import sacc
from sacc import Sacc, standard_types
import twopoint
from twopoint import Types
import sys

input_sacc_filename = sys.argv[1]
output_2pt_filename = sys.argv[2]

# Set real to False if the input file is in Fourier space
real = True
fourier = not real

# Load in the sacc data file from fits method
s = sacc.Sacc.load_fits(input_sacc_filename)

# This table converts the type codes used in sacc to twopoint names
types = {
    standard_types.galaxy_shear_xi_plus: (Types.galaxy_shear_plus_real, Types.galaxy_shear_plus_real),
    standard_types.galaxy_shear_xi_minus: (Types.galaxy_shear_minus_real, Types.galaxy_shear_minus_real),
    standard_types.galaxy_shearDensity_xi_t: (Types.galaxy_position_real, Types.galaxy_shear_plus_real),
    standard_types.galaxy_density_xi: (Types.galaxy_position_real, Types.galaxy_position_real),
    standard_types.galaxy_shear_cl_ee: (Types.galaxy_shear_emode_fourier, Types.galaxy_shear_emode_fourier),
    standard_types.galaxy_shear_cl_bb: (Types.galaxy_shear_bmode_fourier, Types.galaxy_shear_bmode_fourier),
    standard_types.galaxy_shearDensity_cl_e: (Types.galaxy_shear_emode_fourier, Types.galaxy_position_fourier),
    standard_types.galaxy_density_cl: (Types.galaxy_position_fourier, Types.galaxy_position_fourier),
}

# Use builder class from twopoint to build file interating overdata points.

def choose_kernel_and_bins(tracer):
    name, index = tracer.split("_")
    index = int(index) + 1
    if name == 'lens':
        return 'NZ_LENS', index
    elif name == 'source':
        return 'NZ_SOURCE', index
    else:
        raise ValueError(f"Not sure how to convert tracer {tracer} to a 2pt kernel index")

# The 2pt builder object is used to gradually build up a 2pt object from
# data points
builder = twopoint.SpectrumCovarianceBuilder()

# Now we add the data points one by one
for dt in s.get_data_types():
    data = s.get_data_points(dt)
    angbin = 0
    ang_array = [0]
    for d in data:
        stype1 = types[d.data_type][0]
        stype2 = types[d.data_type][1]

        #In sacc we make a new tracer for each bin whereas in 
        # TwoPoint n(z) kernels are grouped into general types. 
        kernel1, bin1 = choose_kernel_and_bins(d.tracers[0])
        kernel2, bin2 = choose_kernel_and_bins(d.tracers[1])

        # Sacc tags store things like angle indices
        if 'xi' in d.data_type:
            #real space
            ang = d['theta']
        if 'cl' in d.data_type:
            # Fourier space
            ang = d['ell']
        print(ang)
        # This relies on the specific ordering in the sacc file to work.
        # The data points have to be in ascending order, and then each
        # time they're not it indicates the start of a new bin.
        if ang < ang_array[-1]: 
            angbin = 1
        else:
            angbin +=1
        ang_array.append(ang)

        # accumulate the data points
        builder.add_data_point(kernel1, kernel2, stype1, stype2, bin2, bin1, ang, angbin, d.value)
        
# Define names of the spectra.  These are our choice, but the selection
# below matches the usage in most DES CosmoSIS pipelines.   

if real:
    names = {
        builder.types[0]: "wtheta",
        builder.types[1]: "gammat",
        builder.types[2]: "xim",
        builder.types[3]: "xip"
    }

if fourier:
    names = {
        builder.types[0]: "galaxy_density_cl",
        builder.types[1]: "galaxy_shearDensity_cl_e",
        builder.types[2]: "galaxy_shear_cl_ee",
    }

builder.set_names(names)

# Load and add covariance
covmat = s.covariance.covmat

# Bulding spectra and conv matrix objects
spectra, covmat_info = builder.generate(covmat, "arcmin")


# Create the number density objects not created using the builder class
nzs_l = []
nzs_s = []

# We combine together the different sacc tracers
# to make one grouped one in 2pt
for tracername in s.tracers:
    tracer = s.get_tracer(tracername)
    if 'lens' in tracername:
        z_l = tracer.z
        nzs_l.append(tracer.nz)
    if 'source' in tracername:
        z_s = tracer.z
        nzs_s.append(tracer.nz)

def zlow(z):
    return z - (z[1] - z[0])/2.

def zhigh(z):
    return z + (z[1] - z[0])/2.

nz_lens = twopoint.NumberDensity('NZ_LENS', zlow(z_l), z_l, zhigh(z_l), nzs_l)
nz_source = twopoint.NumberDensity('NZ_SOURCE', zlow(z_s), z_s, zhigh(z_s), nzs_s)


# Combine together all the bits we've created into a two-point object.
T = twopoint.TwoPointFile(spectra, [nz_lens, nz_source], 
    windows=None, covmat_info=covmat_info)

# And finally save to disc.
T.to_fits(output_2pt_filename, overwrite=True)

