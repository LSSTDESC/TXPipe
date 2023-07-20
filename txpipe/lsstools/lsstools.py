"""
Classes and functions to help computing lss tools 
(primarily within the lss weights stages)
"""
import numpy as np 

class DensityCorrelation:

	def __init__(self,tomobin=None): #add fracdet option here
		"""
		Class to compute and keep track of
		1d density correlations for a given galaxy sample/tomographic bin
		"""

		self.tomobin = tomobin
		self.map_index = np.array([])
		self.smin =  np.array([])
		self.smax =  np.array([])
		self.smean =  np.array([])
		self.N =  np.array([])         #number of galaxies
		self.npix = np.array([])       #number of pixels
		self.ndens_perpixel =  np.array([]) #number density per healpix pixel
		self.norm = np.array([])            #normalization (could be slightly different per map due to outliers)
		self.ndens =  np.array([])          #normalized number density

		self.ndens_err = None
		self.covmat = None

		self.ndens_models = {}
		self.chi2 = {}

		self.mapnames = {} #dict of map names to be indexed with map_index

		self.precomputed_array = {}
		self.precomputed_npix = {}
		self.precomputed_sumsys = {}


	def add_correlation(self, map_index, edges, sys_vals, data, map_input=False, frac=None, sys_name=None, use_precompute=False):
		"""
		add a 1d density correlation

		Params
		------
		map_index: Integer
			For labelling
		edges: 1D array
			edges of the systematic map binning
		sys_vals: 1D array
			systematic map for each fo the valid pixels
		data: 1D array
			systematic value for each object in the galaxy catalog OR the data map pixel values
		map_input: bool
			if True  will assume data is the map pixel values
			if False will assume data is the sys value evaluate at each object location
		frac: 1D array or None
			fractional pixel coverage or each pixel in sys_vals
		sys_name: str
			a label for this map
		use_precompute: bool
			If True will use pre-computed boolean arrays and pixel number counts for this SP map
			If False will compute the density in bins using numpy histogram
		"""
		if sys_name is not None:
			self.mapnames[map_index] = sys_name

		if use_precompute:
			#use the precomputed boolean arrays to get 1d trends

			assert len(self.precomputed_array[map_index][0]) == len(sys_vals)
			assert len(self.precomputed_array[map_index]) == len(edges)-1
			assert map_input
			nobj_sys = np.zeros(len(self.precomputed_array[map_index]))
			for i, select_sp in enumerate(self.precomputed_array[map_index]):
				nobj_sys[i] = np.sum(data[select_sp])

			npix_sys = self.precomputed_npix[map_index]
			sumsys_sys = self.precomputed_sumsys[map_index]

		else:
			#use the numpy histogram functions to get 1d trends

			if frac is None:
				frac = np.ones(len(sys_vals))

			#number counts
			if map_input:
				nobj_sys,_ = np.histogram(sys_vals,bins=edges,weights=data)
			else:
				nobj_sys,_ = np.histogram(data,bins=edges)

			#pixel counts (weighted by frac coverage)
			npix_sys,_ = np.histogram(sys_vals,bins=edges,weights=frac)

			#sumof sys value * frac in sys bin (to make mean)
			sumsys_sys,_ = np.histogram(sys_vals,bins=edges,weights=sys_vals*frac)

		
		#do we want to include any excluded outliers in this normalization?
		norm = np.ones(len(edges)-1)*1./(np.sum(nobj_sys)/np.sum(npix_sys))

		self.map_index = np.append(self.map_index, np.ones(len(edges)-1)*map_index )
		self.smin =  np.append(self.smin, edges[:-1])
		self.smax =  np.append(self.smax, edges[1:])
		self.smean =  np.append(self.smean, sumsys_sys/npix_sys)
		self.N =  np.append(self.N, nobj_sys)
		self.npix = np.append(self.npix, npix_sys)
		self.ndens_perpixel =  np.append(self.ndens_perpixel, nobj_sys/npix_sys )
		self.norm = np.append(self.norm, norm)
		self.ndens =  np.append(self.ndens, norm*nobj_sys/npix_sys )

	def precompute(self, map_index, edges, sys_vals, frac=None):
		"""
		Precompute the boolean arays for SP map bins
		These can be used later for faster computation of 1d trends

		Params
		------
		map_index: Integer
			For labelling
		edges: 1D array
			edges of the systematic map binning
		sys_vals: 1D array
			systematic map for each fo the valid pixels
			The order of these pixels should be preserved in any later calculations
		"""
		if frac is None:
			frac = np.ones(len(sys_vals))

		sp_select = []
		for i_sp_bin in range(len(edges)-1):
			if i_sp_bin == len(edges)-2: #last bin
				sp_select_bin = (sys_vals >= edges[i_sp_bin])*(sys_vals <= edges[i_sp_bin+1]) #this is to match the behaviour of np.histogram
			else:
				sp_select_bin = (sys_vals >= edges[i_sp_bin])*(sys_vals < edges[i_sp_bin+1])
			sp_select.append(sp_select_bin)
		self.precomputed_array[map_index] = np.array(sp_select)

		#pixel counts (weighted by frac coverage)
		npix_sys,_ = np.histogram(sys_vals,bins=edges,weights=frac)

		#sumof sys value * frac in sys bin (to make mean)
		sumsys_sys,_ = np.histogram(sys_vals,bins=edges,weights=sys_vals*frac)
		
		self.precomputed_npix[map_index] = npix_sys
		self.precomputed_sumsys[map_index] = sumsys_sys

	def set_precompute(self, density_correlation):
		"""
		copy over the precompute quantities from a different DensityCorrelation object
		"""
		self.precomputed_array = density_correlation.precomputed_array
		self.precomputed_npix = density_correlation.precomputed_npix
		self.precomputed_sumsys = density_correlation.precomputed_sumsys


	def add_diagonal_shot_noise_covariance(self,assert_empty=True):
		"""
		Adds a shot noise only covariance
		Only adds to the diagonal (no shot noise in the covariance between maps)
		"""
		if assert_empty:
			assert self.ndens_err is None
			assert self.covmat is None
			self.covmat = np.zeros((len(self.N),len(self.N)))

		N_err = np.sqrt(self.N)
		self.ndens_err = self.norm*N_err/self.npix
		covmat_new = np.identity(len(self.N))
		np.fill_diagonal(covmat_new, self.ndens_err**2.)

		self.covmat = self.covmat + covmat_new


	def add_external_covariance(self, covmat, assert_empty=True):
		"""
		Adds an external covariance matrix
		"""
		if assert_empty:
			assert self.ndens_err is None
			assert self.covmat is None
			self.covmat = np.zeros((len(self.N),len(self.N)))

		self.covmat = self.covmat+covmat
		self.ndens_err = np.sqrt(covmat.diagonal())

	def clear_covariance(self):
		self.ndens_err = None
		self.covmat = None

	def get_covmat_singlemap(self, map_index):
		select_map = (self.map_index == map_index)
		return np.array([line[select_map] for line in self.covmat[select_map] ]) 

	def plot1d_singlemap(self, filepath, map_index):
		import matplotlib.pyplot as plt

		select_map = (self.map_index == map_index)
		smean = self.smean[select_map]
		ndens = self.ndens[select_map]

		fig, ax = plt.subplots()
		ax.axhline(1,color='k',ls='--')
		if self.ndens_err is None:
			ax.plot(smean, ndens,'.')
		else:
			ndens_err = self.ndens_err[select_map]
			ax.errorbar(smean, ndens, ndens_err, fmt='.')

		for model_name in self.ndens_models:
			ndens_model = self.ndens_models[model_name][select_map]
			chi2 = self.chi2[model_name][map_index]
			ndata = len(ndens)
			legend_label = model_name+": "+r'$\chi^2=$'+'{0}/{1}'.format(np.round(chi2,1), ndata)
			ax.plot(smean, ndens_model,'-',label=legend_label)

		if map_index in self.mapnames.keys():
			xlabel=self.mapnames[map_index]
		else:
			xlabel=f'SP {map_index}'
		ax.set_xlabel(xlabel)
		ax.set_ylabel(r"$n_{\rm gal}/n_{\rm gal \ mean}$")
		ax.legend()
		fig.savefig(filepath)
		fig.clear()
		plt.close()

	def get_edges(self, map_index):
		"""get the sys map bin edges for a given map"""
		select_map = (self.map_index == map_index)
		smin = self.smin[select_map]
		smax = self.smax[select_map]
		edges = np.zeros(len(smin)+1)
		edges[:-1] = smin
		edges[-1] = smax[-1]
		return edges 

	def add_model(self, model, model_name):
		self.ndens_models[model_name] = model

		#if we have error bars, loop over survey property maps and compute the chi2 for each
		if self.ndens_err is not None:
			self.chi2[model_name] = {}
			for map_index in self.mapnames:
				select_map = (self.map_index == map_index)
				ndens = self.ndens[select_map]
				covmat = self.get_covmat_singlemap(map_index)
				chi2_map = self.calc_chi2(ndens, covmat, model[select_map])
				self.chi2[model_name][map_index] = chi2_map


	@staticmethod
	def calc_chi2(y, err, yfit , v = False):
		if err.shape == (len(y),len(y)):
			#use full covariance
			if v == True:
				print('cov_mat chi2')
			inv_cov = np.linalg.inv( np.matrix(err) )
			chi2 = 0
			for i in range(len(y)):
				for j in range(len(y)):
					chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i,j]*(y[j]-yfit[j])
			return chi2
			
		elif err.shape == (len(y),):
			if v == True:
				print('diagonal chi2')
			return sum(((y-yfit)**2.)/(err**2.))
		else:
			raise IOError('error in err or cov_mat input shape')


def linear_model(beta, *alphas, density_correlation=None, sys_maps=None, sysmap_table=None):
	"""
	linear contamination model:
	F(s) = alpha1*s1 + alpha2*s2 + ... + beta
	Normalised to <F(s)> = 1

	returns
		predicted ndens vs smean
	"""
	import healsparse as hsp
	import healpy as hp
	import numpy as np 
	from . import lsstools

	assert len(alphas) == len(sys_maps)

	if sysmap_table is None:
		sysmap_table = hsplist2array(sys_map)

	#make empty F(s) map
	validpixels = sys_maps[0].valid_pixels
	F = hsp.HealSparseMap.make_empty(sys_maps[0].nside_coverage, sys_maps[0].nside_sparse, dtype=np.float64, sentinel=hp.UNSEEN)
	Fvals = np.sum(np.array([alphas]).T*sysmap_table,axis=0) + beta
	F.update_values_pix(validpixels, Fvals)

	#normalize the map
	meanF = np.mean(F[validpixels]) #TODO make this a weighted mean?
	F.update_values_pix(validpixels, F[validpixels]/meanF)

	data_vals = F[validpixels]
	F_density_corrs = lsstools.DensityCorrelation()
	F_density_corrs.set_precompute(density_correlation)
	for imap, sys_vals in enumerate(sysmap_table):
			edges = density_correlation.get_edges(imap)
			F_density_corrs.add_correlation(imap, edges, sys_vals, data_vals, map_input=True, use_precompute=True )

	return F, F_density_corrs


def calc_chi2(y, cov, yfit):
	import numpy as np 

	inv_cov = np.linalg.inv( np.matrix(cov) )

	chi2 = 0
	for i in range(len(y)):
		for j in range(len(y)):
			chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i,j]*(y[j]-yfit[j])

	return chi2

