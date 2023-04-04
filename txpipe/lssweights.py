from .base_stage import PipelineStage
from .map_correlations import TXMapCorrelations
from .utils import choose_pixelization
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

	This is an abstract base class, which other subclasses
    inherit from to use the same basic structure, which is:
    	- load and process sytematic (survey property) maps
    	- compute 1d density correlations+covariance
    	- compute weights with a regression method
	"""
	name = "TXLSSweights"
	parallel = True
	inputs = [
		("binned_lens_catalog", HDFFile),
		("mask", MapsFile),
	]

	outputs = [
		("lss_weight_summary", FileCollection), #output files and summary statistics will go here
		("lss_weight_maps", MapsFile), #the systematic weight maps to be applied to the lens galaxies
		("binned_lens_catalog_weighted", HDFFile), #the lens catalog with weights added
	]

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.01,
	}

	def run(self):
		pixel_scheme = choose_pixelization(**self.config)
		self.pixel_metadata = pixel_scheme.metadata

		#get number of tomographic lens bins
		with self.open_input("binned_lens_catalog", wrapper=False) as f:
			self.Ntomo = f['lens'].attrs["nbin_lens"] 

		#load the SP maps, apply the mask, normalize the maps (as needed by the method)
		sys_maps, sys_names, sys_meta = self.prepare_sys_maps()

		Fmap_list = []
		for ibin in range(self.Ntomo):

			#compute density vs SP map data vector
			density_corrs = self.calc_1d_density(ibin, sys_maps, sys_names=sys_names)

			#compute covariance of data vector
			covmat = self.calc_covariance(density_corrs) #will need to change the argument to this

			#compute the weights
			Fmap, coeff, coeff_cov = self.compute_weights(density_corrs, sys_maps)
			Fmap_list.append(Fmap)

			#make summary stats and plots
			self.summarize(density_corrs)

		#save object weights and weight maps
		self.save_weights(Fmap_list)


	def read_healsparse(self, map_path, nside):
		"""
		returns a healsparse object degraded to nside
		"""
		import healsparse
		import healpy

		# Convert to correct res healpix map
		m = healsparse.HealSparseMap.read(map_path)
		return m.degrade(nside)

	def prepare_sys_maps(self):
		"""
		By default prepare sysmaps will just load and mask the maps

		subclasses for differnet methods can modify this behaviour 
		(e.g. adding a normalization of the maps)
		"""
		sys_maps, sys_names = self.load_and_mask_sysmaps()
		sys_meta = {'masked':True}
		return sys_maps, sys_names, sys_meta


	def load_and_mask_sysmaps(self):
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

	def calc_1d_density(self, tomobin, sys_maps, sys_names=None):
		"""
		compute the 1d density correlations for a single tomographic lens bin

		Params
		------
		tomobin: Integer
			index for the tomographic lens bin 
		sys_maps: list of Healsparse objects
			list of systematic maps
		sys_names: list of strings, optional
			list of systematic map labels (for labeling plots)

	    Returns
    	-------
    	density_corrs: lsstools.DensityCorrelation 
    		DensityCorrelation instance containing the number 
    		counts/density vs sysmap for all sysmaps
		"""
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

		#pixel ID for each lens galaxy
		obj_pix = hp.ang2pix(nside,ra,dec,lonlat=True, nest=True)

		density_corrs = lsstools.DensityCorrelation() #keeps track of the 1d plots
		for imap, sys_map in enumerate(sys_maps):
			sys_vals = sys_map[sys_map.valid_pixels] #SP value in each valid pixel
			sys_obj = sys_map[obj_pix] #SP value for each object in catalog

			edges = scipy.stats.mstats.mquantiles(sys_vals, percentiles)

			sys_name = None if sys_names is None else sys_names[imap]

			density_corrs.add_correlation(imap, edges, sys_vals, sys_obj, sys_name=sys_name)

		f = time.time()
		print("calc_1d_density took {0}s".format(f-s))

		return density_corrs

	def save_weights(self, Fmap_list):
		"""
		save the weights maps and galaxy weights

		Params
		------
		Fmap_list: list of Healsparse maps
			list of the F(s) maps. one for each tomo bin
			F(s) = 1/weight

		"""
		import healpy as hp

		#### save the weights maps
		map_output_file = self.open_output("lss_weight_maps", wrapper=True)
		map_output_file.file.create_group("maps")
		map_output_file.file["maps"].attrs.update(self.config)

		for ibin, Fmap in enumerate(Fmap_list):
			pix = Fmap.valid_pixels
			map_data = 1./Fmap[pix]
			map_output_file.write_map(f"weight_map_bin_{ibin}", pix, map_data, self.pixel_metadata)


		#### save the binned lens samples
		#There is probably a better way to do this using the batch writer 
		binned_output = self.open_output("binned_lens_catalog_weighted", parallel=True)
		with self.open_input("binned_lens_catalog") as binned_input:
			binned_output.copy(binned_input["lens"],"lens")

		for ibin in range(self.Ntomo):
			#get weight per lens object
			subgroup = binned_output[f"lens/bin_{ibin}/"]
			ra = subgroup["ra"][:]
			dec = subgroup["dec"][:]
			pix = hp.ang2pix(self.pixel_metadata['nside'],ra,dec,lonlat=True,nest=True) #can switch to using txpipe tools for this?
			obj_weight = 1./Fmap[pix]

			subgroup["weight"][...] *= obj_weight

		binned_output.close()

	def summarize(self, density_correlation):
		"""
		make 1d density plots and other summary statistics and save
		
		Params
		------
		density_correlation: lsstools.DensityCorrelation 
		"""
		import numpy as np
		output_dir = self.open_output("lss_weight_summary", wrapper=True)

		for imap in np.unique(density_correlation.map_index):
			density_correlation.plot1d_singlemap(output_dir, imap )

		#add other summary stats here (chi2 tables, best fit coefficients, etc)


class TXLSSweightsSimReg(TXLSSweights):
	"""
	Class compute LSS systematic weights using simultanious linear regression

	Model: 		Linear 
	Covarinace: Shot noise (for now), no correlation between 1d correlations
	Fit: 		Simultaniously fits all sysmaps. By calculating a total  weight map 
				and calculating Ndens vs sysmap directly

	"""
	name = "TXLSSweightsSimReg"
	parallel = True
	inputs = [
		("binned_lens_catalog", HDFFile),
		("mask", MapsFile),
	]

	outputs = [
		("lss_weight_summary", FileCollection), #output files and summary statistics will go here
		("lss_weight_maps", MapsFile), #the systematic weight maps to be applied to the lens galaxies
		("binned_lens_catalog_weighted", HDFFile), #the lens catalog with weights added
	]

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.05,
	}

	def prepare_sys_maps(self):
		"""
		For this method we need sys maps to be normalized to mean 0
		"""
		sys_maps, sys_names = self.load_and_mask_sysmaps()

		#normalize sysmaps (and keep track of the normalization factors)
		mean, std = self.normalize_sysmaps(sys_maps)
		sys_meta = {'masked':True,'mean':mean,'std':std}
		
		return sys_maps, sys_names, sys_meta

	@staticmethod
	def normalize_sysmaps(sys_maps):
		"""
		normalize a list of healsparse maps
		"""
		import numpy as np 

		vpix = sys_maps[0].valid_pixels
		means = []
		stds = []
		for sys_map in sys_maps:
			sys_vals = sys_map[vpix]
			mean = np.mean(sys_vals)
			std = np.std(sys_vals)
			sys_map.update_values_pix(vpix, (sys_vals-mean)/std)
			means.append(mean)
			stds.append(std)

		return np.array(means), np.array(stds) #return means and stds to help reconstruct the original maps later

	def calc_covariance(self, density_correlation):
		"""
		Shot noise-only covariance
		Does not include sample variance terms
		"""
		density_correlation.add_shot_noise_covariance()
		return density_correlation.covmat

	def compute_weights(self, density_correlation, sys_maps ):
		"""
		least square fit to a simple linear model
	
		The function being optimized is a sum of the F(s) maps for each sys map
		
		Params
		------
		density_correlation: lsstools.DensityCorrelation
		sys_maps: list of Healsparse maps

		Returns
		------
		Fmap: Healsparse map
			Map of the fitted function F(s) 
			where F(s) = 1/weight
		coeff: 1D array
			best fit parameters
			coeff=[beta, alpha1, alpha2, etc]
		coeff_cov: 2D array
			covariance matrix of coeff from fit

		"""
		import scipy.optimize

		#we should add an option here to select only the significant trends

		#The linear fit model to be fit 
		#x is a dummy parameter that isn't actually used in the fit
		#There is probably a better way to do this...
		def linear_model_(x,beta,*alpha):
			F, Fdc = linear_model(x,beta,*alpha,
				density_correlation=density_correlation, 
				sys_maps=sys_maps)
			return Fdc.ndens

		#initial parameters
		p0 = [1.0]+[0.1]*len(sys_maps)

		coeff, coeff_cov = scipy.optimize.curve_fit(
		    linear_model_,
		    density_correlation.smin, #x is a dummy parameter for this function
		    density_correlation.ndens,
		    p0=p0,
		    sigma=density_correlation.covmat,
		    absolute_sigma=True,
		    )

		#add the best fit model to this DensityCorrelation instance
		density_correlation.add_model(linear_model_(density_correlation.smin,*coeff))

		#best fit map
		Fmap, _ = linear_model(density_correlation.smin,*coeff,
				density_correlation=density_correlation, 
				sys_maps=sys_maps)

		return Fmap, coeff, coeff_cov
		

def linear_model(x, beta, *alphas, density_correlation=None, sys_maps=None, returnF=False):
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

	return F, F_density_corrs











