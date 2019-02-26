#
#   MCMOLE3D.PY
#   classes to simulate molecular clouds in Milky Way
#
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


import h5py as h5
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import sys
from scipy.stats import norm,gaussian_kde,uniform
from  scipy import histogram2d
import astropy.units  as u
import astropy.coordinates  as coord
from .utils import log_spiral_radial_distribution2

class bash_colors:
    """
    This class contains the necessary definitions to print to bash
    screen with colors. Sometimes it can be useful...
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def header(self,string):
        return self.HEADER+str(string)+self.ENDC
    def blue(self,string):
        return self.OKBLUE+str(string)+self.ENDC
    def green(self,string):
        return self.OKGREEN+str(string)+self.ENDC
    def warning(self,string):
        return self.WARNING+str(string)+self.ENDC
    def fail(self,string):
        return self.FAIL+str(string)+self.ENDC
    def bold(self,string):
        return self.BOLD+str(string)+self.ENDC
    def underline(self,string):
        return self.UNDERLINE+str(string)+self.ENDC

class Cloud(object):
	"""
	Object encoding the properties of each cloud, i.e.:

	- :math:`(R,\phi, z )` the Galactic coordinates of the center of the cloud, in units of :math:`(kpc, rad, kpc)`;
	- :math:`(d_{\odot},\ell, b )` the position referred in the Solar system frame in units of :math:`(kpc, rad,rad)`;
	- :math:`\epsilon`, the emissivity in the center in units of :math:`K km/s`
	- :math:`L` the cloud size in units of :math:`pc`
	"""
	def assign_sun_coord(self,d,latit,longit):

		self.has_suncoord=True
		self.X_sun=[d,latit,longit]
		pass
	def emissivity(self,R):
		"""
		replicating the profile in  eq.(3) of `Puglisi+ 2017 <http://arxiv.org/abs/1701.07856>`_
		"""
		A 	=	globals()['emiss_params'][0]
		R0  =	globals()['emiss_params'][1]
		return A*np.exp(-R/R0)

	def print_cloudinfo(self):
		if not self.has_suncoord:
			print("%d \t %g \t %g \t %g \t %g \t %g\n"%(self.id,self.X[0],self.X[1],self.X[2],self.W,self.L))
		elif self.has_suncoord:
			print("%d \t %g \t %g \t %g \t %g \t %g\t %g \t %g \t %g \n"%(self.id,self.X[0],self.X[1],self.X[2],self.W,self.L,\
																	self.X_sun[0],self.X_sun[1]*180./np.pi,self.X_sun[2]*180./np.pi))
		pass
	def size_function(self,R):
		"""
		Compute the size in a very wide range of cloud sizes  :math:`[s_1,s_2] pc`, from random numbers following
		the probability distribution function coming from the cloud size function eq.(4) of
		`Puglisi+ 2017 <http://arxiv.org/abs/1701.07856>`_.
		We split it  in two parts :

		- if :math:`L>s_0`  a decreasing power-law distribution has been assumed with spectral index :math:`a_L=0.8` (see Heyer Dame 2015).
		- Otherwise  In the outer Galaxy a lower spectral index has been measured  :math:`a_L=3.3`, whereas in the inner Galaxy a steeper one  :math:`a_L=3.9`.

		"""
		s1=globals()['L1']
		s0,s2=globals()['L0'],globals()['L2']
		alpha1=0.8

		if R<8.3:
			alpha2=3.9
		else :
			alpha2=3.3

		#normalization constant such that Integral(dP)=1 in [sizemin,sizemax]
		k2=1./(  1./(alpha1 + 1.) *s1**(-alpha1-alpha2) *( s1**(1+alpha1 )- s0**(1+alpha1) ) \
			+ 1./(1.- alpha2 )* (s2**(1-alpha2)- s1**(1-alpha2)))
		k1=s1**(-alpha1-alpha2) * k2
		X10	=	k1/(alpha1+1.)*(s1**(1+alpha1 )- s0**(1+alpha1))
		X21	=	k2/(1.-alpha2)*(s2**(1-alpha2)- s1**(1-alpha2))

		x=np.random.uniform()
		if x<X10:
			return ((alpha1+1.)/k1 * x + (s0)**(1+alpha1))**(1/(1+alpha1))
		else :
			return  ((1-alpha2)/k2 * (x-X10)  + (s1)**(1-alpha2))**(1/(1-alpha2))

	def __init__(self,idcloud,x1,x2,x3,size=None,em=None):
		self.id=idcloud
		self.X=[x1,x2,x3]
		self.has_suncoord=False

		if (size is None) or (em is  None):
			self.W=self.emissivity(x1)
			self.L=self.size_function(x1)
		else:
			self.L=size
			self.W=em

class Cloud_Population(object):
	"""
	class encoding  a list of `Clouds` objects. It is initialized by setting the total
	number of clouds and the geometry you may want to distribute them.
	It contains `N_clouds` :class:`Cloud` objects.
	"""

	def cartesianize_coordinates(self,array):
		"""
		convert arrays of cylindrical ( spherical if a Spherical distribution is chosen  ) coordinates to cartesian ones.
		It exploits :mod:`astropy` routines.
		"""

		return coord.Galactocentric(array)

	def __call__(self):

		self.clouds=[]
		np.random.seed(11 +self.random_seed)

		a=np.random.uniform(low=0.,high=1.,size=self.n)
		self.phi=2.*np.pi*a

		if self.model=='Spherical':
			self.r= norm.rvs(loc=self.R_params[0],scale=self.R_params[1],size=self.n)

			v=np.random.uniform(low=0.,high=1.,size=self.n)
			self.theta=np.arccos(2.*v-1.)

			coord_array=coord.PhysicsSphericalRepresentation(self.phi*u.rad,self.theta * u.rad,self.r*u.kpc )
			self.cartesian_galactocentric= self.cartesianize_coordinates(coord_array)
			self.heliocentric_coordinates()

			for i,x,p,t,d,latit,longit in zip(np.arange(self.n),self.r,self.phi,self.theta,self.d_sun,self.lat,self.long):
				if x<=0.:
					self.r[i]=np.random.uniform(low=0.,high=1.,size=1)
					x=self.r[i]
				c=Cloud(i,x,p,t,size=None,em=None)
				c.assign_sun_coord(d,latit,longit)
				self.clouds.append(c)

		else:
			self.r=self.phi*0.
			rbar=self.R_params[2]

			if self.model=='Axisymmetric':
				np.random.seed(self.random_seed + 29)
				self.r= norm.rvs(loc=self.R_params[0],scale=self.R_params[1],size=self.n)
				negs=np.ma.masked_less(self.r,0.)
				#central molecular zone
				self.r[negs.mask]=0.
			elif 	self.model=='LogSpiral':
				#the bar is assumed axisymmetric and with an inclination angle phi0~25 deg as
				#it has been measured by  Fux et al. 1999

				phi_0=np.deg2rad(25.)
				self.phi+=phi_0
				subsize=np.int( self.n/10 )
				self.r[0:subsize]=norm.rvs(loc=self.R_params[0],scale=self.R_params[1],size=subsize)
				#np.random.uniform(low=0.,high=8.,size=self.n/4)
				rscale=rbar/1.5
				self.r[subsize:self.n],self.phi[subsize:self.n]=log_spiral_radial_distribution2(\
																rbar,phi_0,self.n-subsize,self.R_params[0],self.R_params[1])
				#self.r[subsize:self.n]=log_spiral_radial_distribution(self.phi[subsize:self.n],rbar,phi_0)
				#simulate the bar
				arr=np.ma.masked_less(self.r,rbar)
				self.r[arr.mask]=abs(np.random.normal(loc=0.,scale=rscale,size=len(self.r[arr.mask])))
				negs=np.ma.masked_less(self.r,0.)
				#central molecular zone
				self.r[negs.mask]=0.

			#the thickness of the Galactic plane is function of the Galactic Radius roughly as ~ 100 pc *cosh((x/R0) ), with R0~10kpc
			# for reference see fig.6 of Heyer and Dame, 2015
			sigma_z0=self.z_distr[0]
			R_z0=self.z_distr[1]

			sigma_z=lambda R: sigma_z0*np.cosh((R/R_z0))
			self.zeta=self.phi*0.
			np.random.seed(self.random_seed + 19)
			for i,x,p in zip(np.arange(self.n),self.r,self.phi):
				self.zeta[i]=np.random.normal(loc=0.,scale=sigma_z(x) )
				self.clouds.append(Cloud(i,x,p,self.zeta[i],size=None,em=None))
			coord_array=coord.CylindricalRepresentation(self.r*u.kpc,self.phi*u.rad,self.zeta*u.kpc )
			self.cartesian_galactocentric = self.cartesianize_coordinates(coord_array)
			self.heliocentric_coordinates()

			for c,d,latit,longit in zip(self.clouds,self.d_sun,self.lat,self.long):
				c.assign_sun_coord(d,latit,longit)
		self.L=np.array(self.sizes)
		self.healpix_vecs=self.compute_healpix_vec()
		self.W=self.get_pop_emissivities_sizes()[0]


	def compute_healpix_vec(self):
		"""
		convert galactic latitude and longitude positions  in :mod:`healpy` mapping quantinties: position vectors.
		"""
		rtod=180./np.pi
		b_h=np.pi/2. - self.lat
		l_h=self.long
		vec=hp.ang2vec(b_h,l_h)

		return vec
	def get_pop_emissivities_sizes(self):
		"""
		Looping into the clouds of this class it returns the emissivity and size of each cloud.
		"""
		sizes,emiss=[],[]
		for c in self.clouds:
			sizes.append(c.L)
			emiss.append(c.W)
		return  emiss, sizes
	def heliocentric_coordinates(self):
		"""
		convert Galactocentric coordinates to heliocentric ones
		"""
		g=self.cartesian_galactocentric.transform_to(coord.Galactic)

		self.d_sun=g.distance.kpc
		self.lat=g.b.rad
		self.long=g.l.rad
	def initialize_cloud_population_from_output(self,filename):
		"""
		read from an hdf5 output file the cloud catalog and assign values
		"""
		self.clouds=[]
		self.read_pop_fromhdf5(filename)

		if self.models[self.model]==1:
			zipped=list(zip(np.arange(self.n),self.r,self.phi,self.theta,self.L,self.W,self.d_sun,self.lat,self.long))
		elif self.models[self.model]>=2:
			zipped=list(zip(np.arange(self.n),self.r,self.phi,self.zeta,self.L,self.W,self.d_sun,self.lat,self.long))
		for i,r,p,t,l,w,d,latit,longit in zipped:
			c=Cloud(i,r,p,t,size=l,em=w)
			c.assign_sun_coord(d,latit,longit)
			self.clouds.append(c)
			c=None
		pass
	def plot_histogram_population(self,figname=None):
		"""
		Makes histograms of all over the population of clouds to check the probability
		density functions (PDF) of the coordinates :math:`R_{gal}, z, d_{\odot}, \ell`.

		.. note::

			Set `figname` to the path of your file where  to  save the plot, otherwise  it outputs to screen.

		"""
		h,edges=np.histogram(self.r,bins=200,normed=True)
		bins=np.array([(edges[i]+edges[i+1])/2. for i in range(len(h))])
		area=np.array([(edges[i+1]-edges[i])*h[i] for i in range(len(h))])
		fig=plt.figure(figsize=(12,9))

		#plt.subplot(2,3,1)
		plt.subplot(2,2,1)
		h,bins,p=plt.hist(self.r,200,normed=True,histtype='stepfilled',alpha=0.3,label='Bin =0.1 kpc')
		ax1 = plt.gca()
		plt.xlim([0,12])
		xmajorLocator = MultipleLocator(2)
		xminorLocator = MultipleLocator(.5)
		xmajorFormatter = FormatStrFormatter('%d')
		ax1.xaxis.set_major_locator(xmajorLocator)
		ax1.xaxis.set_major_formatter(xmajorFormatter)
		ax1.xaxis.set_minor_locator(xminorLocator)

		#ymajorFormatter = FormatStrFormatter('%1')
		#ax.yaxis.set_major_formatter(ymajorFormatter)
		plt.xlabel(r'$R_{gal}\,$ [kpc]',fontsize=17)

		plt.legend(loc='upper right', numpoints = 1,prop={'size':15} )
		#plt.ylabel('N per bin '+r'$(\times 10^5)$',fontsize=17)
		plt.ylabel('PDF',fontsize=17)
		radtodeg=180./np.pi
		plt.subplot(2,2,4)
		if self.model=='Spherical':
			plt.hist(np.cos(self.theta),bins=np.linspace(-1.,1.,5),histtype='stepfilled',alpha=0.3)
			plt.xlabel(r'$\cos(\theta )\, $ ',fontsize=20)
		else:
			plt.hist(self.zeta*1e3,80,normed=True,histtype='stepfilled',alpha=0.3,label='Bin = 5 pc')
			plt.legend(loc='upper right', numpoints = 1,prop={'size':15} )
			plt.xlabel('Vertical position [pc] ',fontsize=17)
			plt.xlim([-200,200])
			ax2 = plt.gca()
			plt.ylim([0,0.01])
			xmajorLocator = MultipleLocator(100)
			xminorLocator = MultipleLocator(10)
			xmajorFormatter = FormatStrFormatter('%d')
			ax2.xaxis.set_major_locator(xmajorLocator)
			ax2.xaxis.set_major_formatter(xmajorFormatter)
			ax2.xaxis.set_minor_locator(xminorLocator)
			ax2.yaxis.set_major_locator(MultipleLocator(.005))
			ax2.yaxis.set_minor_locator(MultipleLocator(.001))


			ymajorFormatter = FormatStrFormatter('%1.1e')
			#ax.yaxis.set_major_formatter(ymajorFormatter)

		plt.subplot(2,2,3,sharey=ax1)
		plt.hist(self.d_sun,200,normed=True,histtype='stepfilled',alpha=0.3,label='Bin =0.1 kpc')
		plt.legend(loc='upper right', numpoints = 1,prop={'size':15} )
		plt.xlabel('Heliocentric Distance [kpc]',fontsize=17)
		plt.xlim([0,20])
		ax3 = plt.gca()
		xmajorLocator = MultipleLocator(5)
		xminorLocator = MultipleLocator(1)
		xmajorFormatter = FormatStrFormatter('%d')
		ax3.xaxis.set_major_locator(xmajorLocator)
		ax3.xaxis.set_major_formatter(xmajorFormatter)
		ax3.xaxis.set_minor_locator(xminorLocator)

		ymajorFormatter = FormatStrFormatter('%1.1e')
		#ax.yaxis.set_major_formatter(ymajorFormatter)
		#plt.ylabel('N per bin '+r'$(\times 10^5)$',fontsize=17)
		plt.ylabel('PDF',fontsize=17)


		plt.subplot(2,2,2)
		m=np.where(self.long >=np.pi)[0]
		l=self.long*0.
		l=self.long
		l[m]=self.long[m] - 2*np.pi
		plt.hist(l*radtodeg,bins=np.linspace(-180,180,72),normed=True,histtype='stepfilled',alpha=0.3,label='Bin = 5 deg ')
		ax = plt.gca()
		xmajorLocator = MultipleLocator(100)
		xminorLocator = MultipleLocator(10)
		xmajorFormatter = FormatStrFormatter('%d')
		ax.xaxis.set_major_locator(xmajorLocator)
		ax.xaxis.set_major_formatter(xmajorFormatter)
		ax.xaxis.set_minor_locator(xminorLocator)
		ax.yaxis.set_major_locator(MultipleLocator(.005))
		ax.yaxis.set_minor_locator(MultipleLocator(.001))
		ymajorFormatter = FormatStrFormatter('%1.1e')
		#ax.yaxis.set_major_formatter(ymajorFormatter)

		plt.xlabel('Galactic Longitude [deg] ',fontsize=17)

		plt.legend(loc='upper right', numpoints = 1,prop={'size':15} )
		plt.xlim([180,-180])
		plt.ylim([0,0.02])

		if figname is None:
			plt.show()
		else:
			plt.savefig(figname)
		plt.close()

	def plot_radial(self,X,ylabel,figname=None,color='b'):
		"""
		Plot a quantity `X` which may variates across the Galactic radius :math:`R_{gal}`.
		(e.g.  the midplane thickness, the emissivity profile,etc...)

		.. note::

			Set `figname` to the path of your file where  to  save the plot, otherwise  it outputs to screen.
		"""
		plt.plot(self.r,X,color+'-')
		plt.xlabel(r'$R_{gal}\, \mathrm{[kpc]}$ ',fontsize=20)
		plt.ylabel(ylabel)
		plt.yscale('log')
		if figname is None:
			plt.show()
		else:
			plt.savefig(figname)
		plt.close()

	def plot_3d_population(self,figname=None):
		"""
		Makes density contour plots of all the cloud population.

		.. note::

			Set `figname` to the path of your file where  to  save the plot, otherwise  it outputs to screen.
		"""

		from matplotlib import gridspec,colors
		x0	=	self.cartesian_galactocentric.x.value
		x1	=	self.cartesian_galactocentric.y.value
		x2	=	self.cartesian_galactocentric.z.value
		planes={'x-y':[x0,x1],'x-z':[x0,x2],'y-z':[x1,x2]}
		c=1
		fig=plt.figure(figsize=(15,15))
		gs  = gridspec.GridSpec(3, 1 )#width_ratios=[1.5, 2,1.5],height_ratios=[1.5,2,1.5])

		for a in list(planes.keys()):
			x,y=planes[a]
			a1,a2=a.split("-",2)

			xyrange=[[min(x),max(x)],[min(y),max(y)]]
			nybins,nxbins=50,50
			bins=[nybins,nxbins]
			thresh=2#density threshold
			hh, locx, locy = histogram2d(x, y, range=xyrange, bins=[nybins,nxbins])
			posx = np.digitize(x, locx)
			posy = np.digitize(y, locy)
			ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
			hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
			xdat1 = x[ind][hhsub < thresh] # low density points
			ydat1 = y[ind][hhsub < thresh]
			hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
			ax=plt.subplot(gs[c-1])

			ax.set_xlabel(a1+' [kpc]',fontsize=20)
			ax.set_ylabel(a2+' [kpc]',fontsize=20)
			if a2=='z' and self.model!='Spherical':
				im=ax.imshow(np.flipud(hh.T),cmap='jet',vmin=0, vmax=hhsub.max()/2, extent=[-15,15, -1,1],interpolation='gaussian', origin='upper')
				ax.set_yticks((-.5,0,.5))
			else:
				im=ax.imshow(np.flipud(hh.T),cmap='jet',vmin=0, vmax=hhsub.max()/2, extent=np.array(xyrange).flatten(),interpolation='gaussian', origin='upper')
				ax.plot(xdat1, ydat1, '.',color='darkblue')
			c+=1
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im,cax=cbar_ax)
		if figname is None:
			plt.show()
		else:
			plt.savefig(figname)
		plt.close()
	def print_pop(self):
		"""
		Output on screen the whole Monte-Carlo catalogue of molecular clouds
		"""


		cols=bash_colors()
		print(cols.header("###"*40))
		print(cols.blue(cols.bold(str(self.n)+" Clouds simulated assuming a "+self.model+" model\n")))
		if self.d_sun is None:
			if self.model=='Spherical':
				print(cols.green("ID \t R \t\t PHI  \t\t THETA\t\t Emissivity \t Size \n"))
				print(cols.green(" \t[kpc]\t\t[rad]\t\t [rad]\t\t  [K km/s]\t [pc]\t [kpc] \t [deg] \t [deg] \n"))

			else:
				print(cols.green("ID \t R  \t\t PHI\t\t Z \t\t Emissivity\t Size \n"))
				print(cols.green(" \t[kpc]\t\t[rad]\t\t[kpc]\t\t[K km/s]\t[pc]\t[kpc] \t [deg] \t [deg]\n"))
			print(cols.header("---"*40))
		else :
			if self.model=='Spherical':
				print(cols.green("ID \t R \t\t PHI  \t\t THETA\t\t Emissivity \t Size \t\t D_sun \t\t b \t\tl\n"))
				print(cols.green(" \t[kpc]\t\t[rad]\t\t [rad]\t\t  [K km/s]\t [pc]\t\t[kpc] \t\t [deg] \t\t [deg]\n"))
			else:
				print(cols.green("ID \t R  \t\t PHI\t\t Z \t\t Emissivity\t Size \t\t D_sun \t\t b \t\tl\n"))
				print(cols.green(" \t[kpc]\t\t[rad]\t\t[kpc]\t\t[K km/s]\t[pc]\t\t[kpc] \t\t [deg] \t\t [deg]\n"))
			print(cols.header("---"*40))
		for c in self.clouds:
			c.print_cloudinfo()

		pass
	def print_parameters(self):
		"""
		print the parameters set for  the simulation

		"""
		typical_size=globals()['L1']
		minsize,maxsize=globals()['L0'],globals()['L2']
		emissivity=globals()['emiss_params']
		

		cols=bash_colors()
		print(cols.header("###"*20))
		print(cols.blue("Parameters  to MCMole3D"))
		print("---"*20)
		print(cols.green("Model\t\t\t\t....\t"),cols.bold(self.model))
		print(cols.green("#clouds\t\t\t\t....\t"),cols.bold(self.n))
		print(cols.green("Molecular Ring [location,width]\t....\t"),cols.bold("%g,%g\t"%(self.R_params[0],self.R_params[1])),"kpc")
		print(cols.green("Central Midplane Thickness\t....\t"),cols.bold("%g\t"%(self.z_distr[0]*1000)),"pc")
		print(cols.green("Radius of the Galactic Bar \t....\t"),cols.bold("%g\t"%(self.R_params[2])),"kpc")
		print(cols.green("Scale Radius Midplane Thickness\t....\t"),cols.bold("%g\t"%self.z_distr[1]),"kpc")
		print(cols.green("Amplitude Emissivity profile\t....\t"),cols.bold("%g\t"%emissivity[0]),"K km/s")
		print(cols.green("Scale Radius Emissivity profile\t....\t"),cols.bold("%g\t"%emissivity[1]),"kpc")
		print(cols.green("Cloud Typical size\t\t....\t"),cols.bold("%g\t"%typical_size),"pc")
		print(cols.green("Cloud size [min,max]\t\t....\t"),cols.bold("%g,%g\t"%(minsize,maxsize)),"pc")
		print(cols.header("###"*20))
		pass

	def read_pop_fromhdf5(self,filename):
		"""
		Read from an hdf5 file the cloud population. the :class:`Cloud_Population` is thus  initialized by :func:`initialize_cloud_population_from_output`
		"""
		f=h5.File(filename,'r')
		g=f["Cloud_Population"]
		self.r=g["R"][...]
		self.phi=g["Phi"][...]
		self.L=g["Sizes"][...]
		self.W=g["Emissivity"][...]
		coord_array=np.zeros(self.n)
		if self.models[self.model]==1:
			self.theta=g["Theta"][...]
			coord_array=coord.PhysicsSphericalRepresentation(self.phi*u.rad,self.theta * u.rad,self.r*u.kpc )
		elif self.models[self.model]>=2:
			self.zeta=g["Z"][...]
			coord_array=coord.CylindricalRepresentation(self.r*u.kpc,self.phi*u.rad,self.zeta*u.kpc )
		self.d_sun=g["D_sun"][...]
		self.long=g["Gal_longitude"][...]
		self.lat=g["Gal_Latitude"][...]
		self.healpix_vecs=g["Healpix_Vec"][...]

		self.cartesian_galactocentric = self.cartesianize_coordinates(coord_array)
		from utilities.utilities_functions import bash_colors

		cols=bash_colors()
		f.close()
		print(cols.bold("////// \t read from "+filename+"\t ////////"))
		pass
	def set_parameters(self,radial_distr=[5.3,2.5,3],emissivity=[60,3.59236], thickness_distr=[0.1,9.],\
							typical_size=10.,size_range=[0.3,30]):
		"""
		Set key-parameters to the population of clouds.

		- ``radial_distr``:{list}
			:math:`(\mu_R,FWHM_R, R_{bar})` parameters to the  Galactic radius  distribution of the clouds,
			assumed Gaussian. The last parameters is the postition of the bar tip.
			Default  :math:`\mu_R= 5.3 \, ,\sigma_R=FWHM_R/\sqrt(2 \ln 2)= 2.12,\, R_{bar}=3 \, kpc`.
		- ``thickness_distr``:{list}
			:math:`(FWHM_{z,0}, R_{z,0})` parameters to the vertical thickness of the Galactic plane
			increasing in the outer Galaxy with :math:`\sigma(z)=sigma_z(0) *cosh(R/R_{z,0})`.
			Default  :math:`\sigma_{z,0}=0.1 \, , R_{z,0}=9` kpc.
		- ``emissivity``:{list}
			Parameters to the emissivity radial profile, :math:`\epsilon(R)= \epsilon_0 \exp(- R/R_0)`.
			Default Heyer and Dame 2015 values : :math:`\epsilon_0=60\, K km/s,\, R_0=3.6 \, kpc`.
		- ``typical_size``: {scalar}
			Typical size of molecular clouds where we observe the peak in the size distribution function.
			Default :math:`L_0=10` pc.
		"""


		self.R_params=list(radial_distr)
		self.R_params[1]/=np.sqrt(2*np.log(2))
		self.z_distr=list(thickness_distr)
		self.z_distr[0]/=(2.*np.sqrt(2.*np.log(2.)))
		globals()['L1']=typical_size
		globals()['L0'],globals()['L2']=size_range
		globals()['emiss_params']=emissivity
		pass

	def write2hdf5(self,filename):
		"""
		Write onto an hfd5 file the whole catalogue

		"""


		f=h5.File(filename,'w')
		g=f.create_group("Cloud_Population")
		g.create_dataset('Healpix_Vec',np.shape(self.healpix_vecs),dtype=h5.h5t.IEEE_F64BE,data=self.healpix_vecs)
		g.create_dataset('R',np.shape(self.r),dtype=h5.h5t.IEEE_F64BE,data=self.r)
		g.create_dataset('Phi',np.shape(self.phi),dtype=h5.h5t.IEEE_F64BE,data=self.phi)
		if self.models[self.model]==1:
			g.create_dataset('Theta',np.shape(self.theta),dtype=h5.h5t.IEEE_F64BE,data=self.theta)
		elif self.models[self.model]>=2:
			g.create_dataset('Z',np.shape(self.zeta),dtype=h5.h5t.IEEE_F64BE,data=self.zeta)

		g.create_dataset('Sizes',np.shape(L),dtype=h5.h5t.IEEE_F64BE,data=L)
		g.create_dataset('Emissivity',np.shape(W),dtype=h5.h5t.IEEE_F64BE,data=W)
		g.create_dataset('D_sun',np.shape(self.d_sun),dtype=h5.h5t.IEEE_F64BE,data=self.d_sun)
		g.create_dataset('Gal_Latitude',np.shape(self.lat),dtype=h5.h5t.IEEE_F64BE,data=self.lat)
		g.create_dataset('Gal_longitude',np.shape(self.long),dtype=h5.h5t.IEEE_F64BE,data=self.long)

		f.close()
		pass


	def __init__(self, N_clouds,model, randseed=0):
		self.model=model
		self.models={'Spherical':1,'Axisymmetric':2,'LogSpiral':3}
		#it's possible to execute simulations with the same random seed
		self.random_seed=randseed

		if self.models[model]==1:
			self.r,self.theta,self.phi=0,0,0
		elif self.models[model]==2:
			self.r,self.zeta,self.phi=0,0,0
		self.n= N_clouds
		self.d_sun,self.lat,self.long =None,None,None


	@property
	def emissivity(self):
		return self.get_pop_emissivities_sizes()[0]
	@property
	def sizes(self):
		return self.get_pop_emissivities_sizes()[1]


class Collect_Clouds(Cloud_Population):
	"""
	List of `Cloud_population` classes . Read from output
	"""

	def __init__(self, N_pops,model,Ncl=4000,filestring=None):

		super(Collect_Clouds,self).__init__(Ncl,model)
		self.Pops=[]
		#compute the populations
		for i in range(N_pops):
			pop=Cloud_Population(self.n,self.model)
			if filestring is None:
				pop()
			else:
				fname=filestring+'_'+self.model+'_'+str(i)+'.hdf5'
				pop.initialize_cloud_population_from_output(fname)

			self.Pops.append(pop)
			pop=None
		self.concatenate_arrays()



	def concatenate_arrays(self):
		"""
		Concatenate arrays of several :class:`Cloud_Population` objects  to process all of these quantities together

		"""
		self.r=np.concatenate([p.r for p in  self.Pops])
		self.phi=np.concatenate([p.phi for p in  self.Pops])
		self.d_sun=np.concatenate([p.d_sun for p in  self.Pops])
		self.lat=np.concatenate([p.lat for p in  self.Pops])
		self.long=np.concatenate([p.long for p in  self.Pops])
		if self.models[self.model]==1:
			self.theta=np.concatenate([p.theta for p in  self.Pops])
		elif self.models[self.model]>=2:
			self.zeta=np.concatenate([p.zeta for p in  self.Pops])
		self.W=np.concatenate([p.get_pop_emissivities_sizes()[0] for p in  self.Pops])
		self.L=np.concatenate([p.get_pop_emissivities_sizes()[1] for p in  self.Pops])


	def write2hdf5(self,filename):
		"""
		Write onto an hfd5 file the whole catalogue

		"""
		f=h5.File(filename,'w')
		g=f.create_group("Cloud_Population")
		g.create_dataset('Healpix_Vec',np.shape(self.healpix_vecs),dtype=h5.h5t.IEEE_F64BE,data=self.healpix_vecs)
		g.create_dataset('R',np.shape(self.r),dtype=h5.h5t.IEEE_F64BE,data=self.r)
		g.create_dataset('Phi',np.shape(self.phi),dtype=h5.h5t.IEEE_F64BE,data=self.phi)
		if self.models[self.model]==1:
			g.create_dataset('Theta',np.shape(self.theta),dtype=h5.h5t.IEEE_F64BE,data=self.theta)
		elif self.models[self.model]>=2:
			g.create_dataset('Z',np.shape(self.zeta),dtype=h5.h5t.IEEE_F64BE,data=self.zeta)

		g.create_dataset('Sizes',np.shape(self.L),dtype=h5.h5t.IEEE_F64BE,data=self.L)
		g.create_dataset('Emissivity',np.shape(self.W),dtype=h5.h5t.IEEE_F64BE,data=self.W)
		g.create_dataset('D_sun',np.shape(self.d_sun),dtype=h5.h5t.IEEE_F64BE,data=self.d_sun)
		g.create_dataset('Gal_Latitude',np.shape(self.lat),dtype=h5.h5t.IEEE_F64BE,data=self.lat)
		g.create_dataset('Gal_longitude',np.shape(self.long),dtype=h5.h5t.IEEE_F64BE,data=self.long)

		f.close()
		pass
