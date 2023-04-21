"""
Classes and functions to help computing lss tools 
(primarily within the lss weights stages)
"""
import numpy as np 

class DensityCorrelation:

	def __init__(self,tomobin=None): #add fracdet option here
		"""
		Class to compute and keep track of keeps track of 
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

	def add_correlation(self, map_index, edges, sys_vals, data, map_input=False, frac=None, sys_name=None ): #add fracdet option here
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
		"""
		if sys_name is not None:
			self.mapnames[map_index] = sys_name

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


	def add_shot_noise_covariance(self,):
		assert self.ndens_err is None
		assert self.covmat is None

		N_err = np.sqrt(self.N)
		self.ndens_err = self.norm*N_err/self.npix
		self.covmat = np.identity(len(self.N))
		np.fill_diagonal(self.covmat, self.ndens_err**2.)

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





