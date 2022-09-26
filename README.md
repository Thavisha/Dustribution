Mapping the Interstellar Dust Extinction and Density in 3D

The latest results from Dustribution can be viewed and downloaded from https://www2.mpia-hd.mpg.de/homes/tmueller/projects/ThavishaDustDensity/index.html

References:
- Dharmawardena et al., 2022, Astronomy & Astrophysics, Volume 658, id.A166, 30 pp. https://www.aanda.org/articles/aa/full_html/2022/02/aa41298-21/aa41298-21.html
- Dharmawardena et al., accepted MNRAS, 2022

** The method utilises latent variable Gaussian processes combined with variational inference with GP optimisation carried out using the gradient descent algorithm ADAMW. 

** At each optimisation iteration the GP predicts a full log10(density) of the region of interest. This is converted to density and then integrated along each line-of-sight to which we have a input star. This integration gives us the model extinction to each star which is then compared to the observed extinction to each star to derive a likelihood.

** By directly fitting GP to log10(density) we maintain positive density throughout our maps and hence ensure monotonically increasing extinction along lines-of-sight. We also correlate all points of the map with one another at all times and therefore overcome fingers-of-god effect. 

** We use Gaussian Process package GPyTorch (Gardner et al. 2018; https://gpytorch.ai/) and probabilistic programming package Pyro (Bingham et al. 2018; Phan et al. 2019; https://pyro.ai/) in our implementation. 

** Input data needed: lb coordinates, distances+uncertainties and extinctions+uncertainties for a set of stars in the region of interest; lb coordinates of the region of interest - approximately rectangle in l and b 

** We also need to decide on the required resolution of the grid on which to trian the GP on - while we can train on high resolution grids the number of stars within grid cells will dictate the real resolution

** We can predict on a higher resolution than the training grid for visualisation purposes - this does not mean we recover that resolution for the structure though!

** It will also help to have an idea of the size of expected structure to be recoved in pc as well as mean density of region and variation from mean to inititate the GP hyperparameters. We allow the scale lengths to vary in the xyz direction on their own and not forced to be same - this allows us to recover filamentary structure not visible when scale length is forced to be the same in all three xyz directions. 

** This script takes in the input region coordinates and resolutions and first builds the training and predicting grids stored for later. 
Then the training grid is used within the GP along with GP setup parameters to train the GP on the input extinctions 
Once the GP is trained it is used to predict on the prediction grid to out put the 16th,50th,84th percentiles of the extinction and density maps of our region which is saved and ready for visualisation

** File structure (In the approximate order of usage in the process): 
- The run_Gpytorch_Ext_andDens_Pred.py file excutes the full process and calls upon the top level functions/routines held within the topLevel_routines.py file. 
- The threeD_Grid_Build.py buils the train and predict grids from the input region coordinates. 
- The gpy_Initiate.py file caries out the GP training and then from the trained GP predicting the extinction and density maps. The GP class for this is held in the gpyClass_Latent.py file. 
- The integration to the individual sources to derive the model extinction which will be compared to the source extinctions for a likelihood in the GP optimisation is carried out by the dens_integTorch.py file
- The plotPerformance.py file plots the perfomance check plots such as ELBO variation with iteration. 
- The plotStructure.py file plots the GP predicted density and extinction maps. 
- The plotResiduals.py file plots the residuals such as Predicted vs. True extinction and residual histogram.

