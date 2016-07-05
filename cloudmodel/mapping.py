import healpy as hp
import h5py as h5
import numpy as np


def gaussian_apodization(x,d):
	"""
	smooth the emissivity of the cloud with a gaussian , centered in the center of the cloud
	and with sigma defined in such a way that:

	.. math::

		d = 6 \sigma

	where ``d`` is the border of the cloud, i.e. the radius of the cloud.
	"""
#	sigma	=	d/np.sqrt(2.*p*np.log(10))
	sigma	=	d / 6.
	y		= 	(x/(np.sqrt(2)*sigma))**2

	return np.exp(-y)

def cosine_apodization(x,d):
	"""
	smooth the emissivity of the cloud with a cosine, with the following relation:
	.. math::

		\epsilon(x) = \epsilon_0 \cos(\frac{\pi}{2} \frac{x}{d})

	where ``d`` is the border of the cloud, i.e. the radius of the cloud, in such a
	way that the  :math:`\epsilon(d)=0`.
	"""
	return np.cos(x/d*np.pi/2.)

def distance_from_cloud_center(theta,phi,theta_c,phi_c):
	"""
	given a position of one pixel :math:`(\theta,\phi)` within the cloud compute the arclength
	of the pixel from the center, onto a unitary sphere.  by considering scalar products of vectors
	to the points  on the sphere to get the angle :math:`\psi` between them. .
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

def do_healpy_map(Pop,nside,fname,apodization='gaussian'):


	N=Pop.n
	mapcloud=np.zeros(hp.nside2npix(nside))
	sizekpc=Pop.L/1.e3

	for i in xrange(N):
		vec 		=	Pop.healpix_vecs[i]
		angularsize =	sizekpc[i]/Pop.d_sun[i]
		em_c=Pop.W[i]
		theta_c,phi_c= hp.vec2ang(vec)
		listpix=hp.query_disc(nside,vec,angularsize)

		theta_pix,phi_pix=hp.pix2ang(nside, listpix )

		distances= distance_from_cloud_center(theta_pix,phi_pix,theta_c,phi_c)

		if apodization == 'cos':
			profile = cosine_apodization(distances,angularsize)
		if apodization == 'gaussian':
			profile = gaussian_apodization(distances,angularsize)

		mapcloud[listpix]	+= em_c*profile

	hp.write_map(fname,mapcloud)
	return mapcloud
