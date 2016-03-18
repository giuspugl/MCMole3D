import h5py as h5
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np 
import sys
from scipy.stats import norm,gaussian_kde
from  scipy import histogram2d
from utilities.utilities_functions import bash_colors
import astropy.units  as u
import astropy.coordinates  as coord


class Cloud(object):
	def print_cloudinfo(self):
		if not self.has_suncoord:
			print "%d \t %g \t %g \t %g \t %g \t %g\n"%(self.id,self.X[0],self.X[1],self.X[2],self.W,self.L)
		elif self.has_suncoord:
			print "%d \t %g \t %g \t %g \t %g \t %g\t %g \t %g \t %g \n"%(self.id,self.X[0],self.X[1],self.X[2],self.W,self.L,\
																	self.X_sun[0],self.X_sun[1]*180./np.pi,self.X_sun[2]*180./np.pi)
		pass

	def emissivity(self,R): 
		"""
		replicating the profile in Heyer Dame 2015
		"""
		A 	=	60 #200 K km/s looks to be too much
		R0	=	3.59236 #kpc
		return A*np.exp(-R/R0)
	def assign_sun_coord(self,d,latit,longit):
		self.has_suncoord=True
		self.X_sun=[d,latit,longit]
		pass 
	def __init__(self,idcloud,x1,x2,x3,size=None,em=None):
		self.id=idcloud
		self.X=[x1,x2,x3]
		self.has_suncoord=False

		if (size is None) or (em is  None): 
			self.W=self.emissivity(x1)
			self.L=norm.rvs(loc=10., scale=30., size=1)
			while self.L<0:
				self.L=norm.rvs(loc=10., scale=30., size=1)
		else:
			self.L=size
			self.W=em

class Cloud_Population(object):
	""" 
	class w/ a list of `Clouds` objects
	"""

	def cartesianize_coordinates(self,array):
		
		return coord.Galactocentric(array)

	def __call__(self): 

		self.clouds=[]
		self.r= norm.rvs(loc=5.3,scale=2.5/(np.sqrt(2.*np.log(2.))),size=self.n)
		a=np.random.uniform(low=0.,high=1.,size=self.n)
		self.phi=2.*np.pi*a
		if self.model=='Spherical':
			v=np.random.uniform(low=0.,high=1.,size=self.n)
			self.theta=np.arccos(2.*v-1.)

			coord_array=coord.PhysicsSphericalRepresentation(self.phi*u.rad,self.theta * u.rad,self.r*u.kpc )
			self.cartesian_galactocentric= self.cartesianize_coordinates(coord_array)
			self.heliocentric_coordinates()

			for i,x,p,t,d,latit,longit in zip(np.arange(self.n),self.r,self.phi,self.theta,self.d_sun,self.lat,self.long): 
				if x<=0.: 
					self.r[i]=0.
					x=0.
				c=Cloud(i,x,p,t,size=None,em=None)
				c.assign_sun_coord(d,latit,longit)
				self.clouds.append(c)
		elif self.model=='Axisymmetric':
			#the thickness of the Galactic plane is function of the Galactic Radius roughly as ~ 75 pc *exp((x/R0)^2 ), with R0~12kpc
			# for reference see fig.6 of Heyer and Dame, 2015
			fwhm=lambda R: 0.075*np.exp((R/12)**2 )
			self.zeta=self.phi*0.
			for i,x,p in zip(np.arange(self.n),self.r,self.phi): 
				if x<=0.: 
					self.r[i]=0.
					x=0.
				self.zeta[i]=np.random.normal(loc=0.,scale=fwhm(x))
				self.clouds.append(Cloud(i,x,p,self.zeta[i],size=None,em=None))

			coord_array=coord.CylindricalRepresentation(self.r*u.kpc,self.phi*u.rad,self.zeta*u.kpc )
			self.cartesian_galactocentric = self.cartesianize_coordinates(coord_array)
			self.heliocentric_coordinates()

			for c,d,latit,longit in zip(self.clouds,self.d_sun,self.lat,self.long):
				c.assign_sun_coord(d,latit,longit)
	
	def plot_histogram_population(self,figname=None):
		
		h,edges=np.histogram(self.r,bins=200,normed=True)
		bins=np.array([(edges[i]+edges[i+1])/2. for i in range(len(h))])
		area=np.array([(edges[i+1]-edges[i])*h[i] for i in range(len(h))])
		fig=plt.figure(figsize=(15,15))
		plt.xlim([0,12])
		plt.subplot(2,3,1)
		h,bins,p=plt.hist(self.r,200,normed=0,histtype='stepfilled',alpha=0.3,label='Bin =0.1 kpc')
		#import matplotlib.mlab as mlab
		#y = mlab.normpdf( bins, 5.3, 2.5/(np.sqrt(2.*np.log(2.))))
		#plt.plot(bins, y, 'r--', linewidth=1)

		plt.xlabel(r'$R_{gal}\, \mathrm{[kpc]}$ ')
		plt.legend(loc='upper right', numpoints = 1,prop={'size':9} )
		plt.ylabel('# clouds per bin')
		plt.grid(True)
		
		plt.subplot(2,3,2)
		radtodeg=180./np.pi
		plt.hist(self.phi*radtodeg,bins=np.linspace(0.,360,5),histtype='stepfilled',alpha=0.3)
		plt.xlabel(r'$\phi \, \mathrm{[deg]}$ ')
		plt.ylabel('# clouds per bin')
		plt.grid(True)

		plt.subplot(2,3,3)
		if self.model=='Spherical':
			plt.hist(np.cos(self.theta),bins=np.linspace(-1.,1.,5),histtype='stepfilled',alpha=0.3)
			plt.xlabel(r'$\cos(\theta )\, $ ')
			plt.ylabel('# clouds per bin')
			plt.grid(True)
			plt.subplot(2,3,6)

			plt.hist(self.lat*radtodeg,bins=np.linspace(-100,100,40),histtype='stepfilled',alpha=0.3)
			plt.xlabel(r'$b \, \mathrm{[deg]}$ ')
			plt.ylabel('# clouds per bin')
			plt.grid(True)
		elif self.model=='Axisymmetric':
			plt.hist(self.zeta*1.e3,80,histtype='stepfilled',alpha=0.3,label='Bin = 5 pc')
			plt.legend(loc='upper right', numpoints = 1,prop={'size':9} )
			plt.xlabel('Vertical position [pc] ')
			plt.ylabel('# clouds per bin')
			plt.grid(True)

			plt.subplot(2,3,6)
			plt.hist(self.lat*radtodeg,bins=np.linspace(-10,10,40),histtype='stepfilled',alpha=0.3)
			plt.xlabel(r'$b \, \mathrm{[deg]}$ ')
			plt.ylabel('# clouds per bin')
		plt.grid(True)

		plt.subplot(2,3,4)
		plt.hist(self.d_sun,200,histtype='stepfilled',alpha=0.3,label='Bin =0.1 kpc')
		plt.legend(loc='upper right', numpoints = 1,prop={'size':9} )
		plt.xlabel('Heliocentric Distance [kpc]')
		plt.ylabel('# clouds per bin')
		plt.grid(True)
		
		plt.subplot(2,3,5)
		plt.hist(self.long*radtodeg,bins=np.linspace(0,360,72),histtype='stepfilled',alpha=0.3,label='Bin = 5 deg ')
		plt.grid(True)
		#plt.hist( (self.long)*radtodeg,bins=np.linspace(180,360,36),histtype='stepfilled',alpha=0.3,label='Bin = 5 deg ')
		plt.xlabel(r'$ \ell \, \mathrm{[deg]}$ ')
		plt.ylabel('# clouds per bin')
		plt.legend(loc='upper right', numpoints = 1,prop={'size':9} )


		if figname is None:
			plt.show()
		else:
			plt.savefig(figname)
		plt.close()
	
	def plot_radial(self,X,ylabel,figname=None): 
		plt.plot(self.r,X,'.')
		plt.xlabel(r'$R_{gal}\, \mathrm{[kpc]}$ ')
		plt.ylabel(ylabel)
		plt.yscale('log')
		if figname is None:
			plt.show()
		else:
			plt.savefig(figname)
		plt.close()

	def plot_3d_population(self,figname=None):
		from matplotlib import gridspec,colors
		x0	=	self.cartesian_galactocentric.x.value
		x1	=	self.cartesian_galactocentric.y.value
		x2	=	self.cartesian_galactocentric.z.value

		planes={'x-y':[x0,x1],'x-z':[x0,x2],'y-z':[x1,x2]}
		c=1
		fig=plt.figure(figsize=(15,15))
		gs  = gridspec.GridSpec(3, 1 )#width_ratios=[1.5, 2,1.5],height_ratios=[1.5,2,1.5])

		for a in planes.keys(): 
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
		
			im=ax.imshow(np.flipud(hh.T),cmap='jet',vmin=0, vmax=hhsub.max()/2, extent=np.array(xyrange).flatten(),interpolation='gaussian', origin='upper')
			ax.set_xlabel(a1+' [kpc]')
			ax.set_ylabel(a2+' [kpc]')
			if a2=='z' and self.model=='Axisymmetric':
				ax.set_yticks((-.5,0,.5))
			else:
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


	def write2hdf5(self,filename): 
		W,L=self.get_pop_emissivities_sizes()
		healpix_vecs=self.compute_healpix_vec()

		f=h5.File(filename,'w')
		g=f.create_group("Cloud_Population")
		g.create_dataset('Healpix_Vec',np.shape(healpix_vecs),dtype=h5.h5t.IEEE_F64BE,data=healpix_vecs)
		g.create_dataset('R',np.shape(self.r),dtype=h5.h5t.IEEE_F64BE,data=self.r)
		g.create_dataset('Phi',np.shape(self.phi),dtype=h5.h5t.IEEE_F64BE,data=self.phi)
		if self.models[self.model]==1:
			g.create_dataset('Theta',np.shape(self.theta),dtype=h5.h5t.IEEE_F64BE,data=self.theta)
		elif self.models[self.model]==2:
			g.create_dataset('Z',np.shape(self.zeta),dtype=h5.h5t.IEEE_F64BE,data=self.zeta)

		g.create_dataset('Sizes',np.shape(L),dtype=h5.h5t.IEEE_F64BE,data=L)
		g.create_dataset('Emissivity',np.shape(W),dtype=h5.h5t.IEEE_F64BE,data=W)
		g.create_dataset('D_sun',np.shape(self.d_sun),dtype=h5.h5t.IEEE_F64BE,data=self.d_sun)
		g.create_dataset('Gal_Latitude',np.shape(self.lat),dtype=h5.h5t.IEEE_F64BE,data=self.lat)
		g.create_dataset('Gal_longitude',np.shape(self.long),dtype=h5.h5t.IEEE_F64BE,data=self.long)

		f.close()
		pass
	def compute_healpix_vec(self):
		#convert latitude and longitude in healpix mapping 
		rtod=180./np.pi
		b_h=np.pi/2. - self.lat
		l_h=self.long
		vec=hp.ang2vec(b_h,l_h)
		
		return vec
	def read_pop_fromhdf5(self,filename):
		f=h5.File(filename,'r')
		g=f["Cloud_Population"]
		self.r=g["R"][...]
		self.phi=g["Phi"][...]
		self.L=g["Sizes"][...]
		self.W=g["Emissivity"][...]
		if self.models[self.model]==1:
			self.theta=g["Theta"][...]
		elif self.models[self.model]==2: 
			self.zeta=g["Z"][...]
		self.d_sun=g["D_sun"][...]
		self.long=g["Gal_longitude"][...]
		self.lat=g["Gal_Latitude"][...]

		cols=bash_colors()
		print cols.bold("////// \t read from "+filename+"\t ////////")

		pass
	def initialize_cloud_population_from_output(self,filename): 
		self.clouds=[]
		self.read_pop_fromhdf5(filename)

		if self.models[self.model]==1:
			zipped=zip(np.arange(self.n),self.r,self.phi,self.theta,self.L,self.W,self.d_sun,self.lat,self.long)
		elif self.models[self.model]==2:
			zipped=zip(np.arange(self.n),self.r,self.phi,self.zeta,self.L,self.W,self.d_sun,self.lat,self.long)
		for i,r,p,t,l,w,d,latit,longit in zipped: 
			c=Cloud(i,r,p,t,size=l,em=w)
			c.assign_sun_coord(d,latit,longit)
			self.clouds.append(c)
			c=None
		pass 

	def get_pop_emissivities_sizes(self):
		sizes,emiss=[],[]
		for c in self.clouds:
			sizes.append(c.L[0])
			emiss.append(c.W)
		return  emiss, sizes

	
	def print_pop(self):
		cols=bash_colors()
		print cols.header("###"*40)
		print cols.blue(cols.bold(str(self.n)+" Clouds simulated assuming a "+self.model+" model\n"))
		if self.d_sun is None:
			if self.model=='Spherical':
				print cols.green("ID \t R \t\t PHI  \t\t THETA\t\t Emissivity \t Size \n")
				print cols.green(" \t[kpc]\t\t[rad]\t\t [rad]\t\t  [K km/s]\t [pc]\t [kpc] \t [deg] \t [deg] \n")

			elif self.model=='Axisymmetric':
				print cols.green("ID \t R  \t\t PHI\t\t Z \t\t Emissivity\t Size \n")
				print cols.green(" \t[kpc]\t\t[rad]\t\t[kpc]\t\t[K km/s]\t[pc]\t[kpc] \t [deg] \t [deg]\n")
			print cols.header("---"*40)
		else :
			if self.model=='Spherical':
				print cols.green("ID \t R \t\t PHI  \t\t THETA\t\t Emissivity \t Size \t\t D_sun \t\t b \t\tl\n")
				print cols.green(" \t[kpc]\t\t[rad]\t\t [rad]\t\t  [K km/s]\t [pc]\t\t[kpc] \t\t [deg] \t\t [deg]\n")
			elif self.model=='Axisymmetric':
				print cols.green("ID \t R  \t\t PHI\t\t Z \t\t Emissivity\t Size \t\t D_sun \t\t b \t\tl\n")
				print cols.green(" \t[kpc]\t\t[rad]\t\t[kpc]\t\t[K km/s]\t[pc]\t\t[kpc] \t\t [deg] \t\t [deg]\n")
			print cols.header("---"*40)
		for c in self.clouds:
			c.print_cloudinfo()

		pass 
	def __init__(self, N_clouds,model):
		self.model=model
		self.models={'Spherical':1,'Axisymmetric':2}
		if self.models[model]==1:
			self.r,self.theta,self.phi=0,0,0
		elif self.models[model]==2: 
			self.r,self.zeta,self.phi=0,0,0
		self.n= N_clouds
		self.d_sun,self.lat,self.long =None,None,None

	def heliocentric_coordinates(self):
		g=self.cartesian_galactocentric.transform_to(coord.Galactic)
		
		self.d_sun=g.distance.kpc
		self.lat=g.b.rad
		self.long=g.l.rad

		
class Collect_Clouds(Cloud_Population):
	"""
	List of `Cloud_population` classes . Read from output
	"""
	
	def __init__(self, N_pops,model,Ncl=4000,filestring=None):
		
		super(Collect_Clouds,self).__init__(Ncl,model)
		self.Pops=[]
		#compute the populations
		for i in xrange(N_pops): 
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
		self.r=np.concatenate([p.r for p in  self.Pops])
		self.phi=np.concatenate([p.phi for p in  self.Pops])
		self.d_sun=np.concatenate([p.d_sun for p in  self.Pops])
		self.lat=np.concatenate([p.lat for p in  self.Pops])
		self.long=np.concatenate([p.long for p in  self.Pops])
		if self.models[self.model]==1:
			self.theta=np.concatenate([p.theta for p in  self.Pops])
		elif self.models[self.model]==2:
			self.zeta=np.concatenate([p.zeta for p in  self.Pops])




