import healpy as hp
import h5py as h5
import numpy as np 

def do_healpy_map(Pop,nside,fname):
	N=Pop.n
	mapcloud=np.zeros(hp.nside2npix(nside))
	sizekpc=Pop.L/1.e3

	for i in xrange(N):
		vec 		=	Pop.healpix_vecs[i]
		angularsize =	sizekpc[i]/Pop.d_sun[i]

		listpix=hp.query_disc(nside,vec,angularsize)
		mapcloud[listpix]	+= Pop.W[i]
	hp.write_map(fname,mapcloud)
	return mapcloud



