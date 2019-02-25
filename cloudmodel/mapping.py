#
#   MAPPING.PY
#   functions to map simulated molecular clouds in FITS file (interfaces to healpy)
#
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#

import healpy as hp
import h5py as h5
import numpy as np


def gaussian_apodization(x,d):
	"""
	Smooth the emissivity of the cloud with a gaussian profile, centered in the center of the cloud and with sigma defined in such a way that:

	.. math::
		d = 6 \sigma

	where :math:`d` is the border of the cloud, coinciding with size of the cloud.

	"""
	sigma	=	d / 6.
	y		= 	(x/(np.sqrt(2)*sigma))**2

	return np.exp(-y)

def cosine_apodization(x,d):
	"""

	Smooth the emissivity of the cloud with a cosine function, with the following relation:

	.. math::

		\epsilon(x) = \epsilon_c \cos ( frac{\pi}{2} frac{x}{d})

	here :math:`d` is the border of the cloud, i.e. the radius of the cloud, in such a	way that the  :math:`\epsilon(d)=0`.

	"""
	return np.cos(x/d*np.pi/2.)

def distance_from_cloud_center(theta,phi,theta_c,phi_c):
	"""
	given a position of one pixel :math:`(theta,\phi)` within the cloud compute the arclength
	of the pixel from the center, onto a unitary sphere.  by considering scalar products of vectors
	to the points  on the sphere to get the angle :math:`\psi` between them.
	This routine is exploited by :func:`do_healpy_map`.

	see for reference :
	`Arclength on a sphere <http://math.stackexchange.com/questions/231221/great-arc-distance-between-two-points-on-a-unit-sphere>`_
	"""
	cos1	=	np.cos(theta_c)
	sin1	=	np.sin(theta_c)
	cos2	=	np.cos(theta)
	sin2	= 	np.sin(theta)
	cosphi	=	np.cos(phi_c - phi)

	psi 	=	np.arccos( cos1 *cos2 + sin1*sin2 *cosphi )

	return psi

def do_healpy_map(Pop,nside,fname=None,apodization='gaussian',polangle=None,depol_map=None, p=1.e-2,highgalcut=0., verbose=False ):
	"""
	Projects the cloud  population into an :mod:`healpy` map as seen as an observer in the 	solar circle.

	**Parameters**

	- ``Pop`` :
	 	:class:`MCMole3D.Cloud_Population`
	- ``nside``:{int}
		Healpix  grid parameter
	- ``fname``:{str}
		path to the fits file where to store the map
	- ``apodization``:{str}
		profile of the cloud (either `gaussian` or `cos`)
	- ``polangle``:{np.array or map}
		the angle of polarization
	- ``depol_map``:{map}
		the depolarization map due to line of sight effects
	- ``p``: {float}
		polarization fraction ( default 1%)
	- ``highgalcut``: {float}
		angle in radians to exclude clouds at high galactic latitudes,
		`sin(b)<= sin(angle)`;

	.. note::

		if `polangle` is set this routine produces 	even polarization maps, i.e. the Q and U Stokes parameters.

	"""

	N=Pop.n

	if polangle is not None:
		mapcloud=np.zeros(hp.nside2npix(nside)*3).reshape((hp.nside2npix(nside),3))
		cospolangle=np.cos(2*polangle)
		sinpolangle=np.sin(2*polangle)
	else:
		mapcloud=np.zeros(hp.nside2npix(nside))

	sizekpc=Pop.L/1.e3

	sincut=np.sin(highgalcut)
	cloudcount=0
	for i in range(N):
		vec 		=	Pop.healpix_vecs[i]
		angularsize =	sizekpc[i]/Pop.d_sun[i]
		I=Pop.W[i]
		theta_c,phi_c= hp.vec2ang(vec)
		# Exclude HGL clouds  (with too large  angular size, 3 degrees)
		#
		is_hgl =  ( np.sin(theta_c -  angularsize/2.  )< sincut) or \
					( np.sin(theta_c  +   angularsize/2.  )< sincut)

		if  is_hgl  and angularsize > np.deg2rad(3):
			cloudcount+=1
			continue

		listpix=hp.query_disc(nside,vec,angularsize)
		theta_pix,phi_pix=hp.pix2ang(nside, listpix )
		distances= distance_from_cloud_center(theta_pix,phi_pix,theta_c,phi_c)
		if apodization == 'cos':
			profile = cosine_apodization(distances,angularsize)
		if apodization == 'gaussian':
			profile = gaussian_apodization(distances,angularsize)
		if polangle is not None:
			if depol_map is None:
				# polarization angle for the whole cloud from a uniform distribution of
				#	random numbers in [0,pi]
				Q	= p*I *cospolangle[i]
				U	= p*I *sinpolangle[i]
				mapcloud[listpix,0]	=mapcloud[listpix,0] +   (I  *	profile)
				mapcloud[listpix,1]	=mapcloud[listpix,1] +   (Q	 *	profile)
				mapcloud[listpix,2]	=mapcloud[listpix,2] +   (U	 *	profile)
			else:
				# polarization angle from map
				#and geometrical  depolarization map
				mapcloud[listpix,0]	+=(I  *	profile)
				mapcloud[listpix,1]	+=(p*depol_map[listpix]*cospolangle[listpix])*I*profile
				mapcloud[listpix,2]	+=(p*depol_map[listpix]*sinpolangle[listpix])*I*profile

		else:
			mapcloud[listpix]	+= I*profile
	if not fname is None:
		hp.write_map(fname,mapcloud)
	if cloudcount!=0 and verbose :
		print(("Excluded %d clouds at high galactic latitude. (|b|>%g)\n"%(cloudcount,(np.pi/2. - highgalcut)*180./np.pi)))

	return mapcloud
