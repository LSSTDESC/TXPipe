"""
Classes and functions to help computing lss tools 
(primarily within the lss weights stages)
"""
import numpy as np 

class DensityCorrelation:

	def __init__(self): #add fracdet option here
		"""
		Class to compute and keep track of keeps track of 
		1d density correlations for a given galaxy sample/tomographic bin
		"""

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

		self.ndens_model = None

		self.mapnames = {} #dict of map names to be indexed with map_index

	def add_correlation(self, map_index, edges, sys_vals, data, map_input=False, sys_name=None ): #add fracdet option here
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
		"""
		if sys_name is not None:
			self.mapnames[map_index] = sys_name

		#number counts
		if map_input:
			nobj_sys,_ = np.histogram(sys_vals,bins=edges,weights=data)
		else:
			nobj_sys,_ = np.histogram(data,bins=edges)

		#pixel counts
		npix_sys,_ = np.histogram(sys_vals,bins=edges) #weight this by fracdet when implemented

		#sumof sys value in sys bin (to make mean)
		sumsys_sys,_ = np.histogram(sys_vals,bins=edges,weights=sys_vals) #make weighted mean when fracdet implemented

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

	def plot1d_singlemap(self, filedir, map_index):
		import matplotlib.pyplot as plt

		if isinstance(filedir,str):
			filepath = filedir +'/'+ f'sys1D_SP{map_index}.png'
		else:
			filepath = filedir.path_for_file(f"sys1D_SP{map_index}.png")

		select_map = (self.map_index == map_index)
		smean = self.smean[select_map]
		ndens = self.ndens[select_map]

		fig, ax = plt.subplots()
		ax.axhline(1,color='k',ls='--')
		if self.ndens_err is None:
			ax.plot(smean, ndens,'.')
		else:
			ndens_err = self.ndens_err[select_map]
			covmat = self.get_covmat_singlemap(map_index)
			chi2 = self.calc_chi2(ndens, covmat, np.ones(len(ndens)))
			ax.errorbar(smean, ndens, ndens_err, fmt='.', label=r'$\chi^2_{\rm null}=$'+'{0}/{1}'.format(np.round(chi2,1), len(ndens)))

		if self.ndens_model is not None:
			ndens_model = self.ndens_model[select_map]
			ax.plot(smean, ndens_model,'-')

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
		select_map = (self.map_index == map_index)
		smin = self.smin[select_map]
		smax = self.smax[select_map]
		edges = np.zeros(len(smin)+1)
		edges[:-1] = smin
		edges[-1] = smax[-1]
		return edges 

	def add_model(self, model):
		self.ndens_model = model

		#add chi2 calculation here too?

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





