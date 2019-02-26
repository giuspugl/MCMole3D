# MCMOLE3D  TUTORIAL

## Package Dependencies
Please install the following packages  before running `MCMole3D`
- `numpy`
- `healpy`
- `h5py`
- `scipy`
- `astropy`

## Initializing cloud population

The first thing to do is to set the geometry model `Axisymmetric, Spherical, LogSpiral` in the `string` format and the number of clouds i.e. `N`. To initialize a cloud population with default parameters:

`Pop=Cloud_Population(N,model)
`
>Note that if you may want to run **several MC realizations** of population of clouds you have to change the random seed during the initialization of each cloud population:
`Pop=Cloud_Population(N,model,randseed=True)`


To simulate one realization of Cloud population with the default parameters:

`Pop()
`

>to **change** the simulation parameters, run before calling `Pop()` :
`Pop.set_parameters(typical_size=L0 ,size_range=[sizemin,sizemax],radial_distr=[Rgal,sigmaring,Rbar],emissivity=[epsilon_c,R_em])`

to see the histogram distribution:

`Pop.plot_histogram_population() `

and the density contour plot:

`Pop.plot_3d_population()`

finally save the clouds into a `HDF5` file catalogue:

`Pop.write2hdf5(filename)`


The class `Cloud_Population` can be initialized from output just after having called the constructor `Pop=Cloud_Population(N,model)`:

`Pop.initialize_cloud_population_from_output(filename)`

## Projecting clouds into a `HEALPIX` map
set the `Healpix` grid parameter `nside` among one of the possible values [see healpy website](http://healpy.readthedocs.io/en/latest/). The map is  saved into a `.fits` file.

`mapcloud=do_healpy_map(Pop,nside,filename.fits)`
`

To compare the simulated maps and the observations we compute the integral in a longitudinal strip  along the Galactic plane, (for further references see [Puglisi et al 2017](http://arxiv.org/abs/1701.07856) ).

<aside class="warning">
The simulations and observation  maps have to be in the same pixel format:  MCMole3D simulation maps are stored choosing a  `ring Healpix` ordering, whereas the Planck maps are released in the `Nested` ordering, you have to reorder one of them to the other's ordering.
See the [Healpix website](http://healpix.jpl.nasa.gov) and its `ud_grade` routine  for further readings about it.
</aside>

`Itot,I_l=integrate_intensity_map(map,nside)`

To **rescale** the simulations to the observation we compute the ratio `f=Itot_sim/Itot_obs`   and divide the simulation maps by this factor.
Once  the integral for both the simulations and the observations  have been computed and plot them with:

`plot_intensity_integrals(I_obs_l,I_sim_l)`

Finally to convolve the maps  to an instrumental beam you can use the `healpy.smoothing` routine.
