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
	parallel = False
	inputs = [
		("binned_lens_catalog_unweighted", HDFFile), #this file is used by the stage to compute weights
		("lens_tomography_catalog_unweighted", HDFFile), #this file is copied at the end and a weighted version is made (for stages that use this instead of the binned catalogs)
		("mask", MapsFile),
	]

	outputs = [
		("lss_weight_summary", FileCollection), #output files and summary statistics will go here
		("lss_weight_maps", MapsFile), #the systematic weight maps to be applied to the lens galaxies
		("binned_lens_catalog", HDFFile), #the lens catalog with weights added
		("lens_tomography_catalog", HDFFile), #the tomography file with weights added
	]

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.01,
	}

	def run(self):
		"""
		Follows the basic stucture of a regression based weights pipeline:

		(1) Prepare survey properties (SP maps) load, degrade, normalize, etc 
		(2) Compute 1d density trends Ngal vs SP
		(3) Compute the covariance matrix of the density trends
		(4) Compute the weights using a given model
		(5) Summarize (save plots, regression params, etc)
		(6) Save the weighted catalog and weight maps

		Each step can be overwritten in sub-classes
		"""
		pixel_scheme = choose_pixelization(**self.config)
		self.pixel_metadata = pixel_scheme.metadata
		
		#check the metadata nside matches the mask (might not be true of you use an external mask)
		with self.open_input("mask", wrapper=True) as map_file:
			mask_nside = map_file.read_map_info("mask")["nside"]
		assert self.pixel_metadata['nside'] == mask_nside

		#get number of tomographic lens bins
		with self.open_input("binned_lens_catalog_unweighted", wrapper=False) as f:
			self.Ntomo = f['lens'].attrs["nbin_lens"] 

		#output directory for the plots and summary stats
		output_dir = self.open_output("lss_weight_summary", wrapper=True)

		#load the SP maps, apply the mask, normalize the maps (as needed by the method)
		sys_maps, sys_names, sys_meta = self.prepare_sys_maps()

		Fmap_list = []
		for ibin in range(self.Ntomo):

			#compute density vs SP map data vector
			density_corrs = self.calc_1d_density(ibin, sys_maps, sys_names=sys_names)

			#compute covariance of data vector
			covmat = self.calc_covariance(density_corrs, sys_maps) #will need to change the argument to this

			#compute the weights
			Fmap, fit_output = self.compute_weights(density_corrs, sys_maps)
			Fmap_list.append(Fmap)

			#make summary stats and plots
			self.summarize(output_dir, density_corrs, fit_output)

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

	def get_deltag(self, tomobin):
		"""
		convert the ra, dec of the lens sample to pixel number counts
		"""
		import collections
		import healpy as hp
		import numpy as np  
		import healsparse as hsp

		with self.open_input("mask", wrapper=True) as map_file:
			nside = map_file.read_map_info("mask")["nside"]
			nest = map_file.read_map_info("mask")["nest"]
			mask = map_file.read_map("mask")

		#load the ra and dec of this lens bins
		with self.open_input("binned_lens_catalog_unweighted", wrapper=False) as f:
			ra = f[f"lens/bin_{tomobin}/ra"][:]
			dec = f[f"lens/bin_{tomobin}/dec"][:]

		#pixel ID for each lens galaxy
		obj_pix = hp.ang2pix(nside,ra,dec,lonlat=True, nest=True)

		count = collections.Counter(obj_pix)
		pixel = np.array(list(count.keys()))
		Ncounts = np.array(list(count.values())).astype(np.float64)

		if nest:
			maskpix = np.where(mask!=hp.UNSEEN)[0]
		else:
			maskpix = hp.ring2nest(nside, np.where(mask!=hp.UNSEEN)[0])

		#fractional coverage
		frac = hsp.HealSparseMap.make_empty(32, nside, dtype=np.float64)
		frac.update_values_pix(maskpix, mask[np.where(mask!=hp.UNSEEN)[0]])

		deltag = hsp.HealSparseMap.make_empty(32, nside, dtype=np.float64)
		deltag.update_values_pix(maskpix, 0.0)
		deltag.update_values_pix( pixel, Ncounts )
		n = deltag[maskpix]/frac[maskpix]
		nmean = np.average(n, weights=frac[maskpix])
		deltag.update_values_pix( maskpix, n/nmean-1.0 ) #overdenity map 

		return deltag, frac


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
			mask_nest = map_file.read_map_info("mask")["nest"]
			mask = hsp.HealSparseMap(
				nside_coverage=32, 
				healpix_map=(mask==hp.UNSEEN).astype('int'), 
				nest=mask_nest, sentinel=0)
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

		return np.array(sys_maps), np.array(sys_names)

	def calc_1d_density(self, tomobin, sys_maps, sys_names=None):
		"""
		compute the binned 1d density correlations for a single tomographic lens bin

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
			nest = map_file.read_map_info("mask")["nest"]
			mask = map_file.read_map("mask")

		#load the ra and dec of this lens bins
		with self.open_input("binned_lens_catalog_unweighted", wrapper=False) as f:
			ra = f[f"lens/bin_{tomobin}/ra"][:]
			dec = f[f"lens/bin_{tomobin}/dec"][:]
			input_weight = f[f"lens/bin_{tomobin}/weight"][:]

			assert (input_weight==1.).all() # For now lets assume the input weights have to be 1 
											# (we could drop this condition 
											# If we ever want to input a weighted catalog)

		#pixel ID for each lens galaxy
		obj_pix = hp.ang2pix(nside,ra,dec,lonlat=True, nest=True)

		density_corrs = lsstools.DensityCorrelation(tomobin=tomobin) #keeps track of the 1d plots
		for imap, sys_map in enumerate(sys_maps):
			sys_vals = sys_map[sys_map.valid_pixels] #SP value in each valid pixel
			sys_obj = sys_map[obj_pix] #SP value for each object in catalog
			
			if nest: #ideally we dont want if statements like this....
				frac = mask[sys_map.valid_pixels]
			else:
				frac = mask[hp.nest2ring(nside, sys_map.valid_pixels)]

			edges = scipy.stats.mstats.mquantiles(sys_vals, percentiles)

			sys_name = None if sys_names is None else sys_names[imap]

			density_corrs.add_correlation(imap, edges, sys_vals, sys_obj, frac=frac, sys_name=sys_name)

			#also precompute the SP bined arrays and pixel counts
			density_corrs.precompute(imap, edges, sys_vals, frac=frac )

		f = time.time()
		print("calc_1d_density took {0}s".format(f-s))

		return density_corrs

	def save_weights(self, Fmap_list):
		"""
		Save the weights maps and galaxy weights

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
		binned_output = self.open_output("binned_lens_catalog", parallel=True)
		with self.open_input("binned_lens_catalog_unweighted") as binned_input:
			binned_output.copy(binned_input["lens"],"lens")

		for ibin in range(self.Ntomo):
			#get weight per lens object
			subgroup = binned_output[f"lens/bin_{ibin}/"]
			ra = subgroup["ra"][:]
			dec = subgroup["dec"][:]
			pix = hp.ang2pix(self.pixel_metadata['nside'],ra,dec,lonlat=True,nest=True) #can switch to using txpipe tools for this?
			obj_weight = 1./Fmap_list[ibin][pix]
			obj_weight[obj_weight==1./hp.UNSEEN] = 0.0

			subgroup["weight"][...] *= obj_weight


		#### save the tomography file
		tomo_output = self.open_output("lens_tomography_catalog", parallel=True)
		with self.open_input("lens_tomography_catalog_unweighted") as tomo_input:
			tomo_output.copy(tomo_input["tomography"],"tomography")

		lens_weight = tomo_output['tomography/lens_weight'][...]
		for ibin in range(self.Ntomo):
			binned_subgroup = binned_output[f"lens/bin_{ibin}/"]
			mask = (tomo_output['tomography/lens_bin'][...] == ibin)
			lens_weight[mask] *= binned_subgroup["weight"][...]
		tomo_output['tomography/lens_weight'][...] = lens_weight

		#### close the outputs
		tomo_output.close()
		binned_output.close()



	def summarize(self, output_dir, density_correlation, fit_output):
		"""
		make 1d density plots and other summary statistics and save them
		
		Params
		------
		density_correlation: lsstools.DensityCorrelation 
		"""
		import numpy as np
		ibin = density_correlation.tomobin

		#make 1d density plots
		for imap in np.unique(density_correlation.map_index):
			filepath = output_dir.path_for_file(f"sys1D_lens{ibin}_SP{imap}.png")
			density_correlation.plot1d_singlemap(filepath, imap )


		#save fitted map names and IDs
		fitted_maps_id_file = output_dir.path_for_file(f"fitted_map_id_lens{ibin}.txt")
		np.savetxt(fitted_maps_id_file, fit_output['sig_map_index'])
		fitted_maps_names = [density_correlation.mapnames[i] for i in fit_output['sig_map_index']]
		fitted_maps_names_file = output_dir.path_for_file(f"fitted_map_names_lens{ibin}.txt")
		names_file = open(fitted_maps_names_file,'w')
		names_file.write('\n'.join(fitted_maps_names))
		names_file.close()

		#save fitted coefficients 
		coeff_file = output_dir.path_for_file(f"coeff_lens{ibin}.txt")
		np.savetxt(coeff_file, fit_output['coeff'])		

		#save coeff covariances
		if fit_output['coeff_cov'] is not None:
			coeff_cov_file = output_dir.path_for_file(f"coeff_cov_lens{ibin}.txt")
			np.savetxt(coeff_cov_file, fit_output['coeff_cov'])	

		#save chi2 for all models
		for model in density_correlation.chi2.keys():
			chi2_file = output_dir.path_for_file(f"chi2_lens{ibin}_{model}.txt")
			np.savetxt(chi2_file, np.array(list(density_correlation.chi2[model].items())))



class TXLSSweightsSimReg(TXLSSweights):
	"""
	Class compute LSS systematic weights using simultanious linear regression on the binned
	1D correlations

	Model: 		Linear 
	Covarinace: Shot noise (for now), no correlation between 1d correlations
	Fit: 		Simultaniously fits all sysmaps. By calculating a total  weight map 
				and calculating Ndens vs sysmap directly

	"""
	name = "TXLSSweightsSimReg"
	parallel = False

	config_options = {
		"supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
		"nbin": 20,
		"outlier_fraction": 0.05,
		"pvalue_threshold": 0.05, #max p-value for maps to be corrected
		"simple_cov":False, #if True will use a diagonal shot noise only covariance for the 1d relations  
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
		Normalize a list of healsparse maps to mean=0, std=1
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

	def calc_covariance(self, density_correlation, sys_maps):
		"""
		Construct the covariance matrix of the ndens vs SP data-vector 
		"""

		s = time.time()

		#add diagonal shot noise
		density_correlation.add_diagonal_shot_noise_covariance(assert_empty=True)

		if self.config["simple_cov"] == False:
			#add diagonal shot noise
			cov_shot_noise_full = self.calc_covariance_shot_noise_offdiag(density_correlation, sys_maps)
			density_correlation.add_external_covariance(cov_shot_noise_full, assert_empty=False)

			#TO DO: add clustering Sample Variance here

		f = time.time()
		print("calc_covariance took {0}s".format(f-s))

		return density_correlation.covmat

	def calc_covariance_shot_noise_offdiag(self, density_correlation, sys_maps):
		"""
		Shot noise-only covariance (off-diagonal blocks only)
		Does not include sample variance terms
		Off-diagonal terms are between different SP maps
		see https://github.com/elvinpoole/1dcov/blob/main/notes/1d_covariance_notes.pdf
		"""
		import healpy as hp 
		import numpy as np 

		#get nside from the mask
		with self.open_input("mask", wrapper=True) as map_file:
			nside = map_file.read_map_info("mask")["nside"]

		# load the galaxy ra and dec of this lens bins
		# TODO: load lens sample in chunks if needed for memory
		tomobin = density_correlation.tomobin
		with self.open_input("binned_lens_catalog_unweighted", wrapper=False) as f:
			ra = f[f"lens/bin_{tomobin}/ra"][:]
			dec = f[f"lens/bin_{tomobin}/dec"][:]

		#pixel ID for each lens galaxy
		obj_pix = hp.ang2pix(nside,ra,dec,lonlat=True, nest=True)

		#Covariance matrix on number *counts*
		nbinstotal = len(density_correlation.map_index)
		covmat_N = np.zeros((nbinstotal,nbinstotal))
		
		map_list = np.unique(density_correlation.map_index).astype('int')

		#loop over each pair of maps to get 2d histograms
		# and fill in the Nobj covarianace matrix blocks
		map_list = np.unique(density_correlation.map_index).astype('int')
		n2d = {}
		for imap in map_list:
			sys_map_i = sys_maps[imap]
			sys_obj_i = sys_map_i[obj_pix]
			edgesi = density_correlation.get_edges(imap)
			maski = np.where(density_correlation.map_index.astype('int') == imap)[0]
			starti = maski[0]
			finishi = maski[-1]+1
			for jmap in map_list:
				if jmap <= imap:
					continue
				sys_map_j = sys_maps[jmap]
				sys_obj_j = sys_map_j[obj_pix]
				edgesj = density_correlation.get_edges(jmap)
				maskj = np.where(density_correlation.map_index.astype('int') == jmap)[0]
				startj = maskj[0]
				finishj = maskj[-1]+1

				n2d_pair,_,_ = np.histogram2d(sys_obj_i,sys_obj_j,bins=(edgesi,edgesj))
				n2d[imap,jmap] = n2d_pair

				#CHECK I GOT THESE THE RIGHT WAY AROUND!!!!
				covmat_N[starti:finishi,startj:finishj] = n2d_pair
				covmat_N[startj:finishj,starti:finishi] = n2d_pair.T

		#convert N (number count) covariance into n (normalized number density) covariance
		#cov(n1,n2) = cov(N1,N2)*norm**2/(Npix1*Npix2)
		npix1npix2 = np.matrix(density_correlation.npix).T*np.matrix(density_correlation.npix)
		norm2 = np.matrix(density_correlation.norm).T*np.matrix(density_correlation.norm)
		covmat_ndens = covmat_N * np.array(norm2) / np.array(npix1npix2)

		return covmat_ndens


	def select_maps(self, density_correlation):
		"""
		Returns the map indices that have small null p-values (large chi2)
		Threshold p-value set in config
		"""
		import scipy.stats
		import numpy as np 

		chi2_null = density_correlation.chi2['null']
		nbins_per_map = self.config["nbin"]
		map_index_array = np.array(list(chi2_null.keys()))
		chi2_array = np.array(list(chi2_null.values()))

		#pvalues of the null signal for each map
		p = 1.-scipy.stats.chi2(nbins_per_map).cdf(chi2_array)

		sig_maps = map_index_array[p < self.config["pvalue_threshold"]]

		return sig_maps

	def compute_weights(self, density_correlation, sys_maps ):
		"""
		Least square fit to a simple linear model
	
		The function being optimized is a sum of the F(s) maps for each sys map plus a constant
		
		Params
		------
		density_correlation: lsstools.DensityCorrelation
		sys_maps: list of Healsparse maps

		Returns
		------
		Fmap: Healsparse map
			Map of the fitted function F(s) 
			where F(s) = 1/weight
		coeff_output: (sig_map_index, coeff, coeff_cov)
			sig_map_index: 1D array
				index of each map considered significant
			coeff: 1D array
				best fit parameters
				coeff=[beta, alpha1, alpha2, etc]
			coeff_cov: 2D array
				covariance matrix of coeff from fit

		"""
		import scipy.optimize
		import numpy as np
		from . import lsstools

		s = time.time()

		#first add the null signal as the first model
		null_model = np.ones(len(density_correlation.ndens))
		density_correlation.add_model(null_model, 'null')

		#select only the significant trends
		sig_map_index = self.select_maps(density_correlation)
		sys_maps = sys_maps[sig_map_index]
		sysmap_table = hsplist2array(sys_maps)

		#initial parameters
		p0 = np.array([1.0]+[0.0]*len(sys_maps))

		#negative log likelihood to be minimized
		def neg_log_like(params):
			beta = params[0]
			alphas = params[1:]
			F, Fdc = lsstools.lsstools.linear_model(beta,*alphas,
				density_correlation=density_correlation, 
				sys_maps=sys_maps,
				sysmap_table=sysmap_table,)
			chi2 = lsstools.lsstools.calc_chi2(density_correlation.ndens,density_correlation.covmat,Fdc.ndens)
			return chi2/2.

		minimize_output = scipy.optimize.minimize(neg_log_like, p0, method="Nelder-Mead" )
		coeff = minimize_output.x
		coeff_cov = None

		#add the best fit model to this DensityCorrelation instance
		Fmap_bf, Fdc_bf = lsstools.lsstools.linear_model(coeff[0],*coeff[1:],
			density_correlation=density_correlation, 
				sys_maps=sys_maps,
				sysmap_table=sysmap_table,)
		density_correlation.add_model(Fdc_bf.ndens, 'linear')

		#assemble the fitting outputs (chi2, values, coefficents, )
		fit_output = {}
		fit_output['sig_map_index'] = sig_map_index
		fit_output['coeff'] = coeff
		fit_output['coeff_cov'] = coeff_cov

		f = time.time()
		print("compute_weights took {0}s".format(f-s))

		return Fmap_bf, fit_output



class TXLSSweightsLinPix(TXLSSweightsSimReg):
	"""
	Class compute LSS systematic weights using simultanious linear regression at the 
	pixel level using scikitlearn simple linear regression

	Model:				Linear 
	1D Covarinace:		Shot noise (for now), no correlation between 1d correlations
	Pixel Covarinace:	Shot noise, no correlation between pixels
	Fit:				Simultaniously fits all sysmaps using sklearn

	"""
	name = "TXLSSweightsLinPix"
	parallel = False

	def compute_weights(self, density_correlation, sys_maps ):
		"""
		Least square fit to a simple linear model at pixel level 
	
		using p-value of binned 1d trends for regularisation
		
		Params
		------
		density_correlation: lsstools.DensityCorrelation
		sys_maps: list of Healsparse maps

		Returns
		------
		Fmap: Healsparse map
			Map of the fitted function F(s) 
			where F(s) = 1/weight
		coeff_output: (sig_map_index, coeff, coeff_cov)
			sig_map_index: 1D array
				index of each map considered significant
			coeff: 1D array
				best fit parameters
				coeff=[beta, alpha1, alpha2, etc]
			coeff_cov: 2D array
				covariance matrix of coeff from fit

		"""
		import scipy.optimize
		import numpy as np
		import healpy as hp 
		import healsparse as hsp
		from . import lsstools
		from sklearn import linear_model as sk_linear_model

		s = time.time()

		#first add the null signal as the first model
		null_model = np.ones(len(density_correlation.ndens))
		density_correlation.add_model(null_model, 'null')

		#select only the significant trends
		sig_map_index = self.select_maps(density_correlation)
		sys_maps = sys_maps[sig_map_index]
		sysmap_table = hsplist2array(sys_maps)

		#make the delta_g map and fracdet weights
		deltag, frac = self.get_deltag(density_correlation.tomobin)
		dg1 = deltag[sys_maps[0].valid_pixels]+1.0 #deltag+1 for valid pixels
		weight = frac[sys_maps[0].valid_pixels] #weight for samples (prop to 1/sigma^2)

		#do linear regression
		reg = sk_linear_model.LinearRegression()
		reg.fit(sysmap_table.T, dg1, sample_weight=weight )

		#get output of regression
		Fvals = reg.predict(sysmap_table.T)
		Fmap_bf = hsp.HealSparseMap.make_empty(sys_maps[0].nside_coverage, sys_maps[0].nside_sparse, dtype=np.float64, sentinel=hp.UNSEEN)
		Fmap_bf.update_values_pix(sys_maps[0].valid_pixels, Fvals.astype(np.float64) )
		coeff = np.append(reg.intercept_,reg.coef_)
		coeff_cov = None

		#make the 1d trends for the regression output
		Fdc_bf = lsstools.DensityCorrelation()
		Fdc_bf.set_precompute(density_correlation)
		for imap, sys_vals in enumerate(sysmap_table):
			edges = density_correlation.get_edges(imap)
			Fdc_bf.add_correlation(imap, edges, sys_vals, Fmap_bf[Fmap_bf.valid_pixels], map_input=True, use_precompute=True )
		density_correlation.add_model(Fdc_bf.ndens, 'linear')

		#assemble the fitting outputs (chi2, values, coefficents, )
		fit_output = {}
		fit_output['sig_map_index'] = sig_map_index
		fit_output['coeff'] = coeff
		fit_output['coeff_cov'] = coeff_cov

		f = time.time()
		print("compute_weights took {0}s".format(f-s))

		return Fmap_bf, fit_output



def hsplist2array(hsp_list):
	"""
	Convert a list of healsparse maps to a 2d array of the valid pixels
	Assume all maps have the same mask
	"""
	import numpy as np 
	validpixels = hsp_list[0].valid_pixels
	out_array = []
	for i in range(len(hsp_list)):
		out_array.append(hsp_list[i][validpixels])
	return np.array(out_array)



class TXLSSweightsUnit(TXLSSweights):
	"""
	Class assigns weight=1 to all lens objects

	"""
	name = "TXLSSweightsUnit"
	parallel = False

	config_options = { 
	}

	def prepare_sys_maps(self):
		"""
		For unit weights we dont need to load anything
		"""
		sys_maps = sys_names = sys_meta = None
		return sys_maps, sys_names, sys_meta

	def calc_1d_density(self, tomobin, sys_maps, sys_names=None):
		"""
		For unit weights we dont need 1d density trends
		"""
		return None


	def calc_covariance(self, density_correlation, sys_maps):
		"""
		For unit weights we dont need 1d density trends
		"""
		return None

	def summarize(self, output_dir, density_correlation, fit_output):
		"""
		For unit weights we have nothing to summarize
		"""
		pass


	def compute_weights(self, density_correlation, sys_maps ):
		"""
		Creates a healsparse map of unit weights
		Uses the mask directly
		"""
		import numpy as np 
		import healpy as hp 
		import healsparse as hsp

		with self.open_input("mask", wrapper=True) as map_file:
			mask = map_file.read_map("mask")
			nside = map_file.read_map_info("mask")["nside"]
			nest = map_file.read_map_info("mask")["nest"]

		nside_coverage = 32
		nside_sparse = nside

		validpixels = np.where(mask != hp.UNSEEN)[0]
		if nest == False:
			validpixels = hp.ring2nest(nside, validpixels)

		Fmap = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=np.float64, sentinel=hp.UNSEEN)
		Fmap.update_values_pix(validpixels, 1.0)

		fit_output = None
		return Fmap, fit_output

