#
#   UTILIS.PY
#   miscellaneous  functions
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#



import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def plot_2powerlaw_size_function(s0,s1,s2,figname=None):
	"""
	Plot histograms and Probability function of the cloud size function \
	as explained in eq.(4) and (5) of `Puglisi+ 2017 <http://arxiv.org/abs/1701.07856>`_.

	It is defined by three parameters : `s0`,	`s1`,`s2`, being respectively the,\
	  typical cloud size (where the size function peaks), the minimum and the maximum sizes. Thus :math:`s_1<s_0<s_2`.

	  .. note::

		  Set `figname` to the path of your file where  to  save the plot, otherwise  it outputs to screen.

	"""


	alpha1=.8
	spectral=[3.3,3.9]
	coldict=dict.fromkeys(spectral)
	coldict={i:j for i,j in zip(spectral,['blue','red'])}

	for alpha2,col in zip(spectral,['b-','r--']):
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
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.hist(sizes,bins=70,normed=True,alpha=0.4,color=coldict[alpha2])
		plt.yscale('log', nonposy='clip')
		plt.xscale('log')
		plt.ylabel(r'$\xi(L)$',fontsize=20)
		plt.subplot(2,1,2)
		plt.xlim([s0,s2])
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)

		plt.plot(l1,p1(l1),col,label=r'$\alpha_L=$'+str(alpha2))
		plt.plot(l2,p2(l2),col)
		plt.xlabel(r'$L $ [ pc ]',fontsize=20)
		plt.ylabel(r'$\mathcal{P}(<L)$',fontsize=20)
	plt.legend(loc='best',prop={'size':15})
	if not (figname is None):
			plt.savefig(figname)
	plt.show()
	pass

def pixelsize(nside,arcmin=True):
	"""
	Given a `nside` :mod:`healpy` gridding  parameter  returns the pixel size of the chosen pixelization in arcmin (or in radians if `arcmin= False`)
	"""
	if arcmin:
		return np.sqrt(4./np.pi  /hp.nside2npix(nside))*(180*60.)
	else :
		return np.sqrt(4.*np.pi  /hp.nside2npix(nside))

def plot_intensity_integrals(obs_I,mod_I,model=None,figname=None):
	"""
	Plot the intensity integrals computed with :func:`integrate_intensity_map` for both
	 the obseverved  maps(`obs_I`) and the one simulated with MCMole3D `mod_I`.

	 .. note::

		 - Set `figname` to the path of your file where  to  save the plot, otherwise  it outputs to screen.
		 - Set `model` to put the title and the
	"""

	stringn=['observ','model']
	for l,s in zip([obs_I,mod_I],stringn):
		nbins_long=len(l)
		nsteps_long=nbins_long+1
		long_edges=np.linspace(0.,2*np.pi,num=nsteps_long)
		long_centr=[.5*(long_edges[i]+ long_edges[i+1]) for i in range(nbins_long)]

		long_deg=np.array(long_centr)*np.rad2deg(1.)
		longi=np.concatenate([long_deg[nbins_long/2:nbins_long]-360,long_deg[0:nbins_long/2]])
		ob=np.concatenate([l[nbins_long/2:nbins_long],l[0:nbins_long/2]])

		plt.plot(longi,ob,label=r'$I^{'+s+'}(\ell)$')
	plt.yscale('log')
	plt.xlim([-180,180])
	plt.ylim([1.e-1,3.e3])
	plt.ylabel(r'$I(\ell)$ K km/s',fontsize=20)
	plt.xlabel('Galactic Longitude  ',fontsize=20)

	plt.legend(loc='best',prop={'size':15})
	if not model is None:
		plt.title(model+' Model')
	if figname is None:
		plt.show()
	else :
		plt.savefig(figname)

def integrate_intensity_map(Imap,nside,latmin=-2,latmax=2. ,nsteps_long=500,rad_units=False,planck_map=False):
	"""
	Compute the integral of the intensity map along latitude and longitude; to compare observed
	intensity map and the model one.
	To check consistency of the model we compute the integral as in eqs.(6) and (7) of
	`Puglisi+ 2017 <http://arxiv.org/abs/1701.07856>`_.

	*Parameters*

	- `Imap`:{array}
		intensity map
	- `nside`: {int}
		:mod:`healpy` gridding parameter
	- `latmin`, `latmax`:{double}
		minimum and maximum latitudes in `degree` where to perform the integral (default :math:`\pm 2\, deg`)
		if you have the angles in radiants set `rad_units` to `True`.
	- `nsteps_long`:{int}
		number of longitudinal bins, (default 500)
	- `planck_map`:{bool}
		if set to `True`, it sets to zero all  the :mod:`healpy.UNSEEN` masked pixels of the map,
		(useful when dealing with observational maps).

	**Returns**

	- `I_l` :{array}
		latitude integration within the set interval :math:`[b_{min}, b_{max}]`
	- `I_tot`:{double}
		integration of `I_l` in :math:`\ell \in [0,2 \pi]`.

	"""
	if planck_map:
		arr=np.ma.masked_equal(Imap,hp.UNSEEN)
		Imap[arr.mask]=0.


	if not rad_units:
		latmin=np.pi/2.+(np.deg2rad(latmin))
		latmax=np.pi/2.+(np.deg2rad(latmax))

	nbins_long=nsteps_long-1
	long_edges=np.linspace(0.,2*np.pi,num=nsteps_long)
	long_centr=[.5*(long_edges[i]+ long_edges[i+1]) for i in range(nbins_long)]
	listpix=[]
	for i in range(nbins_long):
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

def log_spiral_radial_distribution2(rbar,phi_bar,n,rloc,sigmar):
	"""
	values of pitch angle from `Vallee'2015 <https://arxiv.org/abs/1505.01202>`_  :math:`i=12` deg.

	*Parameters*

	- `rbar`: {float}
		Galactic radius [kpc] where the bar begins
	- `phi_bar`: {float}
		angle :math:`\phi_0` of the bar tip
	- `n`:{int}
		 number of clouds to be distributed following the logspiral geometry
	- `rloc`,`sigmar`:{floats}
		Location and width of  molecular ring in kpc.

	*Return*

	- `r`, `phi`: {arrays}
		array (`n`-size) of  galactic radii and azimut angle following a logspiral distribution
	"""
	pitch=-12.0*np.pi/180.
	pitch2=-12.*np.pi/180.
	pitch3=-12.*np.pi/180.

	Rbar=rbar
	Rmax=12
	theta0= lambda R,A,B: A *(np.log(abs(R))+B) # this will take negative values .... better to put abs()
	radii=	norm.rvs(loc=rloc ,scale=sigmar,size=n)
	phi=radii*0.

	phi[0:np.int(n/4)]=theta0(radii[0:np.int(n/4)],1./np.tan(pitch),-np.log(rbar))  -np.pi +phi_bar
	phi[np.int(n/4):np.int(n/2)]=theta0(radii[np.int(n/4):np.int(n/2)],1./np.tan(pitch2),-np.log(rbar))  -2*np.pi +phi_bar
	phi[np.int(n/2):np.int(3*n/4)]=theta0(radii[np.int(n/2):np.int(3*n/4)],1./np.tan(pitch3),-np.log(rbar))  -np.pi/2. +phi_bar
	phi[np.int(3*n/4):n]=theta0(radii[np.int(3*n/4):n],1./np.tan(pitch),-np.log(rbar))  -3.*np.pi/2. +phi_bar

	r=radii*0.
	sigmamin=.30#kpc

	sigmamax=.4 #kpc
	m=(sigmamax - sigmamin)/(Rmax-Rbar)
	q=sigmamin- m*Rbar

	for i,ir in np.ndenumerate(radii) :
		if ir <Rbar:
			continue
		sigma = abs(m*ir +q)
		r[i]=norm.rvs(loc=ir ,scale=sigma,size=1)

	return r,phi
