from .base_stage import PipelineStage
from .map_correlations import TXMapCorrelations
from .data_types import (
    HDFFile,
    ShearCatalog,
    TextFile,
    MapsFile,
    FileCollection
)
import glob
import time

class TXLSSweights(TXMapCorrelations):
	"""
	BaseClass to compute LSS systematic weights

	Not to be run directly

	This class can be used as a base class for any systematic correction method 
	using regression against survey property maps 
	"""
	name = "TXLSSweights"
	parallel = True
	inputs = [
		("binned_lens_catalog", HDFFile),
		("binned_random_catalog", HDFFile),
		("mask", MapsFile),
	]

	outputs = [
		("lss_weight_output", FileCollection), #output files and summary statistics will go here
		("lss_weights", HDFFile), #the systematic weights to be applied to the lens galaxies
	]

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.05,
	}

	def run(self):

		#get number of tomographic lens bins
		with self.open_input("binned_lens_catalog", wrapper=False) as f:
			Ntomo = f['lens'].attrs["nbin"] - 1 #looks like there is a bug in the binned lens catalog Ntomo so the -1 is temp

		#load the SP maps and apply the mask
		sys_maps, sys_names = self.load_and_mask_sp_maps()

		for ibin in range(Ntomo):

			#compute density vs SP map data vector
			density_corrs = self.calc_1d_density(ibin, sys_maps)

			#compute covariance of data vector
			covmat = self.calc_covariance(density_corrs) #will need to change the argument to this

			#compute the weights
			coeff, cov = self.compute_weights(density_corrs, sys_maps)

			for imap in range(len(sys_maps)):
				density_corrs.plot1d_singlemap(f'./test/sys{imap}.png', imap )



	def read_healsparse(self, map_path, nside):
		"""
		returns a healsparse object degraded to nside
		"""
		import healsparse
		import healpy

		# Convert to correct res healpix map
		m = healsparse.HealSparseMap.read(map_path)
		return m.degrade(nside)

	def calc_1d_density(self, tomobin, sys_maps):
		import scipy.stats
		import healpy as hp
		import numpy as np 
		from . import lsstools

		s = time.time()

		nsysbins = self.config["nbin"]
		f = 0.5 * self.config["outlier_fraction"]
		percentiles = np.linspace(f, 1 - f, nsysbins + 1)

		with self.open_input("mask", wrapper=True) as map_file:
			nside = map_file.read_map_info("mask")["nside"]

		#load the ra and dec of this lens bins
		with self.open_input("binned_lens_catalog", wrapper=False) as f:
			ra = f[f"lens/bin_{tomobin}/ra"][:]
			dec = f[f"lens/bin_{tomobin}/dec"][:]
			input_weight = f[f"lens/bin_{tomobin}/weight"][:]
			assert (input_weight==1.).all() # For now lets assume the input weights have to be 1 
											# (we could drop this condition 
											# If we ever want to input a weighted catalog)

		obj_pix = hp.ang2pix(nside,ra,dec,lonlat=True, nest=True)

		density_corrs = lsstools.DensityCorrelation() #keeps track of the 1d plots

		for imap, sys_map in enumerate(sys_maps):
			sys_vals = sys_map[sys_map.valid_pixels] #SP value in each valid pixel
			sys_obj = sys_map[obj_pix] #SP value for each object in catalog

			edges = scipy.stats.mstats.mquantiles(sys_vals, percentiles)

			density_corrs.add_correlation(imap, edges, sys_vals, sys_obj)

		f = time.time()
		print("calc_1d_density took {0}s".format(f-s))

		return density_corrs


	def load_and_mask_sp_maps(self):
		"""
		load the SP maps and mask them
		"""
		import numpy as np 
		import healpy as hp
		import healsparse as hsp

		s = time.time()

		root = self.config["supreme_path_root"]
		sys_files = glob.glob(f"{root}*.hs")
		nsys = len(sys_files)
		print(f"Found {nsys} total systematic maps")

		with self.open_input("mask", wrapper=True) as map_file:
			mask = map_file.read_map("mask")
			mask = hsp.HealSparseMap(
				nside_coverage=32, 
				healpix_map=(mask==hp.UNSEEN).astype('int'), 
				nest=False, sentinel=0)
			nside = map_file.read_map_info("mask")["nside"]

		#maskpix = np.where(mask!=hp.UNSEEN)[0]

		sys_maps = []
		sys_names = []
		for i, map_path in enumerate(sys_files):
			# strip root, .hs, and underscores to get friendly name
			sys_name = map_path[len(root) : -3].strip("_")
			sys_names.append(sys_name)

			# get actual data for this map
			sys_map = self.read_healsparse(map_path, nside)
			sys_map.apply_mask(mask, mask_bits=1)

			#apply mask
			sys_maps.append(sys_map)

		f = time.time()
		print("load_and_mask_sp_maps took {0}s".format(f-s))

		return sys_maps, sys_names


class TXLSSweightsSimReg(TXLSSweights):
	"""
	Class compute LSS systematic weights using simultanious linear regression
	"""
	name = "TXLSSweightsSimReg"
	parallel = True
	inputs = [
		("binned_lens_catalog", HDFFile),
		("binned_random_catalog", HDFFile),
		("mask", MapsFile),
	]

	outputs = [
		("lss_weight_output", FileCollection), #output files and summary statistics will go here
		("lss_weights", HDFFile), #the systematic weights to be applied to the lens galaxies
	]

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.05,
	}

	def calc_covariance(self, density_correlation):
		"""
		Shot noise-only covariance
		Does not include sample variance terms
		"""
		density_correlation.add_shot_noise_covariance()
		return density_correlation.covmat

	def compute_weights(self, density_correlation, sys_maps ):
		import scipy.optimize

		def linear_model_(x,beta,*alpha):
			return linear_model(x,beta,*alpha,
				density_correlation=density_correlation, 
				sys_maps=sys_maps)

		#initial parameters
		p0 = [1.0]+[0.1]*len(sys_maps)

		coeff, coeff_cov = scipy.optimize.curve_fit(
		    linear_model_,
		    density_correlation.smin, #x is a dummy parameter for this function, this is ugly
		    density_correlation.ndens,
		    p0=p0,
		    sigma=density_correlation.covmat,
		    absolute_sigma=True,
		    )

		density_correlation.ndens_model = linear_model_(density_correlation.smin,*coeff)

		return coeff, coeff_cov
		

def linear_model(x, beta, *alphas, density_correlation=None, sys_maps=None):
	"""
	linear contamination model:
	F(s) = alpha1*s1 + alpha2*s2 + ... + beta

	returns
		predicted ndens vs smean
	"""
	import healsparse as hsp
	import numpy as np 
	from . import lsstools

	assert len(alphas) == len(sys_maps)

	#make empty F(s) map
	validpixels = sys_maps[0].valid_pixels
	F = hsp.HealSparseMap.make_empty(sys_maps[0].nside_coverage, sys_maps[0].nside_sparse, dtype=np.float64)
	F.update_values_pix(validpixels, beta)

	for ialpha, alpha in enumerate(alphas):
		F.update_values_pix(validpixels, F[validpixels]+alpha*sys_maps[ialpha][validpixels])

	F_density_corrs = lsstools.DensityCorrelation()
	for imap, sys_map in enumerate(sys_maps):
			sys_vals = sys_map[sys_map.valid_pixels] #SP value in each valid pixel
			data_vals = F[sys_map.valid_pixels]

			edges = density_correlation.get_edges(imap)
			F_density_corrs.add_correlation(imap, edges, sys_vals, data_vals, map_input=True)

	return F_density_corrs.ndens











