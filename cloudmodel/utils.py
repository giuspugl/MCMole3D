import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def deg2rad(angle):
	return angle*np.pi/180.
def rad2deg(angle):
	return angle*180./np.pi

def plot_size_function(sizemin,sizemax):

	for alpha_L in [3.9,3.3]:
	#normalization constant such that Integral(dP)=1 in [sizemin,sizemax]

		k=(sizemax**(1-alpha_L)- sizemin**(1-alpha_L))
		x=np.random.uniform(size=40000)
		sizes=(k * x + (sizemin)**(1-alpha_L))**(1/(1-alpha_L))
		l=np.linspace(sizemin,sizemax,256)
		p=lambda l: 1./k*(l**(1-alpha_L) - sizemin**(1-alpha_L))
		plt.subplot(2,1,1)
		plt.xlim([8,100])
		plt.hist(sizes,bins=100,normed=False,alpha=0.6)
		plt.yscale('log', nonposy='clip')
		plt.xscale('log')
		plt.ylabel(r'$\xi(L)$')
		plt.subplot(2,1,2)
		plt.xlim([5,50])
		plt.plot(l,p(l),label=r'$\alpha_L=$'+str(alpha_L))

		plt.xlabel(r'$L $ [ pc ]')
		plt.ylabel(r'$\mathcal{P}(<L)$')
	plt.legend(loc='best')
	plt.savefig('/home/peppe/pb2/figures/sizefunction.pdf')
	plt.show()
	pass

def plot_2powerlaw_size_function(s0,s1,s2):

	alpha1=.8
	spectral=[3.3,3.9]
	for alpha2,col in zip(spectral,['b-','g-']):
	#normalization constant such that Integral(dP)=1 in [sizemin,sizemax]

		k2=1./(  1./(alpha1 + 1.) *s1**(-alpha1-alpha2) *( s1**(1+alpha1 )- s0**(1+alpha1) ) \
			+ 1./(1.- alpha2 )* (s2**(1-alpha2)- s1**(1-alpha2)))
		k1=s1**(-alpha1-alpha2) * k2

		X10	=	k1/(alpha1+1.)*(s1**(1+alpha1 )- s0**(1+alpha1))
		X21	=	k2/(1.-alpha2)*(s2**(1-alpha2)- s1**(1-alpha2))

		x=np.random.uniform(size=40000)
		sizes=[]
		for i in x:
			if i<X10:
				sizes.append(((alpha1+1.)/k1 * i + (s0)**(1+alpha1))**(1/(1+alpha1)))
			else :
				sizes.append( ((1-alpha2)/k2 * (i-X10)  + (s1)**(1-alpha2))**(1/(1-alpha2)) )
		l1=np.linspace(s0,s1,64)
		l2=np.linspace(s1,s2,64)
		p1=lambda l: k1/(1+alpha1)*(l**(1+alpha1) - s0**(1+alpha1))
		p2=lambda l: k2/(1-alpha2)*(l**(1-alpha2) - s1**(1-alpha2)) + X10
		plt.subplot(2,1,1)
		plt.xlim([s0,100])
		plt.hist(sizes,bins=70,normed=True,alpha=0.4)
		plt.yscale('log', nonposy='clip')
		plt.xscale('log')
		plt.ylabel(r'$\xi(L)$')
		plt.subplot(2,1,2)
		plt.xlim([s0,s2])
		plt.plot(l1,p1(l1),col,label=r'$\alpha_L=$'+str(alpha2) )
		plt.plot(l2,p2(l2),col)
		plt.xlabel(r'$L $ [ pc ]')
		plt.ylabel(r'$\mathcal{P}(<L)$')
	plt.legend(loc='best')
	plt.savefig('/home/peppe/pb2/figures/sizefunction_2powerlaw.pdf')
	plt.show()
	pass

def pixelsize(nside,arcmin=True):
	if arcmin:
		return np.sqrt(4./np.pi  /hp.nside2npix(nside))*(180*60.)
	else :
		return np.sqrt(4.*np.pi  /hp.nside2npix(nside))

def plot_intensity_integrals(obs_I,mod_I,fname=None):
	nbins_long=len(obs_I)

	nsteps_long=nbins_long+1
	long_edges=np.linspace(0.,2*np.pi,num=nsteps_long)
	long_centr=[.5*(long_edges[i]+ long_edges[i+1]) for i in xrange(nbins_long)]

	long_deg=np.array(long_centr)*rad2deg(1.)
	longi=np.concatenate([long_deg[nbins_long/2:nbins_long]-360,long_deg[0:nbins_long/2]])
	ob=np.concatenate([obs_I[nbins_long/2:nbins_long],obs_I[0:nbins_long/2]])
	mod=np.concatenate([mod_I[nbins_long/2:nbins_long],mod_I[0:nbins_long/2]])

	plt.plot(longi,mod,'--',label=r'$I^{model}(\ell)$')
	plt.plot(longi,ob,label=r'$I^{observ}(\ell)$')
	plt.yscale('log')
	plt.xlim([-180,180])
	plt.ylim([1.e-1,3.e3])
	plt.ylabel(r'$I(\ell)$ K km/s')
	plt.xlabel('Galactic Longitude  ')
	plt.legend(loc='best')
	if fname is None:
		plt.show()
	else :
		plt.savefig(fname)

def integrate_intensity_map(Imap,nside,latmin=-2,latmax=2. ,nsteps_long=500,rad_units=False,planck_map=False):
	"""
	Compute the integral of the intensity map along latitude and longitude; to compare observed
	intensity map and the model one.
	To check consistency of the model we compute:

	.. math::
	\int db d\ell I^{model}(\ell,b) \approx \int db d\ell I^{observ}(\ell,b)
	see (Bronfman et al. 1988).
	"""
	if planck_map:
		arr=np.ma.masked_equal(Imap,hp.UNSEEN)
		Imap[arr.mask]=0.


	if not rad_units:
		latmin=np.pi/2.+(deg2rad(latmin))
		latmax=np.pi/2.+(deg2rad(latmax))

	nbins_long=nsteps_long-1
	long_edges=np.linspace(0.,2*np.pi,num=nsteps_long)
	long_centr=[.5*(long_edges[i]+ long_edges[i+1]) for i in xrange(nbins_long)]
	listpix=[]
	for i in xrange(nbins_long):
		v=[ hp.ang2vec(latmax, long_edges[i]),
			hp.ang2vec(latmax, long_edges[i+1]),
			hp.ang2vec(latmin, long_edges[i+1]),
			hp.ang2vec(latmin, long_edges[i])]
		listpix.append(hp.query_polygon(nside,v))
	delta_b=pixelsize(nside,arcmin=False)
	delta_l=2*np.pi/nbins_long

	I_l=[sum(Imap[l])*delta_b for l in listpix ]
	Itot= sum(I_l)*delta_l
	return Itot,I_l

def log_spiral_radial_distribution(phi,rbar,phi_bar):
	"""
	values of pitch angle from Vallee' 1505.01202 i=13 deg
	"""
	pitch=deg2rad(-15.0)
	pitch2=deg2rad(-13.)

	Rbar=rbar
	arm0=lambda theta: Rbar*np.exp((theta-np.pi)*np.tan(pitch))#**1.5
	arm1=lambda theta: Rbar*np.exp((theta-2*np.pi)*np.tan(pitch2))#**1.5
	arr=np.ma.masked_greater_equal(phi,np.pi+phi_bar)
	less_then_pi=np.logical_not(arr.mask)
	sigma=1.2
	r=phi*0.
	for i in xrange(len(phi)):
		if arr.mask[i]:
			func=arm1
		else:
			func = arm0
		r[i]=norm.rvs(loc=func(phi[i]),scale=sigma,size=1)


	#ax=plt.subplot(111,projection='polar')
	#plt.plot(phi,r,'.')
	#plt.plot(phi[less_then_pi],arm0(phi[less_then_pi]),'.')
	#plt.plot(phi[arr.mask],arm1(phi[arr.mask]),'.')

	#plt.show()

	return r
