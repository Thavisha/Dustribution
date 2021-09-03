import numpy as np
import pandas as pd
import time
import sys
import torch
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian

from topLevel_routines import read_inputTable, reCalc_TrainGrid, checkSource_bounds, reCalc_PredGrid, reTrain_GP, rePredict_GP
from plotStructure import plot_GP_Pred_Dens_Slices_AlongDist, plot_GP_Pred_Ext_Slices_AlongDist, plot_GP_Pred_ExtCumilative
from plotResiduals import plot_Res_ExtHist, plot_Res_Subtract_ExtHist_NormbyUnc
from plotPerformance import plot_PerformanceMetrics


"""

Run file for algorithm - 3D density map of MW regions 

Uses latent varaible GPs combined with variational inference

All files executed by running this top level file 

Input data needed: lb coordinates, distances+uncertainties and extinctions+uncertainties for a set of stars in the region of interest; lb coordinates of the region of interest - a ~rectangle in l and b 

We also need to decide on the required resolution of the grid on which to trian the GP on - while we can train on high resolution grids the number of stars within grid cells will dictate the real resolution

We can predict on a higher resolution than the training grid for visualisation purposes - this does not mean we recover that resolution for the structure though!

It will also help to have an idea of the size of expected structure to be recoved in pc as well as mean density of region and variation from mean to inititate the GP hyperparameters

This script takes in the input region coordinates and resolutions and first builds the training and predicting grids stored for later. 
Then the training grid is used within the GP along with GP setup parameters to train the GP on the input extinctions 
Once the GP is trained it is used to predict on the prediction grid to out put the 16th,50th,84th percentiles of the extinction and density maps of our region which is saved and ready for visualisation

Code run time: GP Run Time 

"""


if __name__=="__main__":


    start_time = time.time()
    print("code run start time = ", start_time)


    ###### Start of input parameters which need to be set by user ######

    #Read in input data from source Table
    input_filename = "orion_catalog_Truncated_withDist_295-550pc.csv"
    source_df = read_inputTable(input_filename)

    #Size of input source sample to condition the GP on - i.e: number of soures to fit the model extinctions on
    subsample_size = len(source_df) #Give number if we only want that number of sources from the full input table


    #Define condition grid boundaries in lbd coordinates - defines the map region and resolution
    #These numbers need to be changed to match the part of the sky which is being mapped 
    l_lower_train = 180.0 #lognitude l in degrees
    l_upper_train = 217.0
    n_l_train = 20 #number of cells within boundaries - defines resolution of grid  

    b_lower_train = -25.5 #latitude b in degrees
    b_upper_train = -3.8
    n_b_train = 20

    d_min_train = 250.0 #Ditance d  in parsecs
    d_max_train = 550.0 
    n_d_train = 25

    #Define predicting grid - enhance visualisation capabilities of grid
    #These numbers need to be changed to match the part of the sky which is being mapped - The l,b,d coordinates must match the train grid coordiantes above. 
    l_lower_pred = 180.0
    l_upper_pred = 217.0
    n_l_pred = 100
   
    b_lower_pred = -25.5
    b_upper_pred = -3.8
    n_b_pred = 100
    
    d_min_pred = 250.0
    d_max_pred = 550.0 
    n_d_pred = 105 


    #GP hyperparameter priors for the RBF Kernel - Setsup the staring point for the hyperparameters - they are allowed to be optimised with the GP
    scale_length_x = 10.0 #Scale length - parsec units - #Approximated from literature given size of structure in region of interest dependent on how close/far we are from the source allowing us to recover the smaller/larger structure. If the region of interest is further away from us we will likely only be able to recover the large scale structure so we need a larger scale length to reflect the size of the large scale structure. But if we are close by we maybe sensitive to the small scale structure as well so we need a smaller scale lenght to reflect the smaller stucture sizes.  
    scale_length_y = 10.0
    scale_length_z = 10.0 
    mean_ext_dens = -3.333 #Mean density - log10(Mags per pc) units - Approximate mean slope of the extinction (units=Mag per pc) - determined from literature
    exp_scalefac = -1.215 #ln(Scale factor) kernel - approximate size of offset expected from the mean -  determined from literature and fit trials


    #Pyro/ADAMW ELBO optimisation paramters
    learning_rate = 0.01 #Scales the size of steps taken when optimising the free paramters with each itteration - Bigger lr means fewer itterations and less likely to get stuck in a local minimum, but if the lr is too small we'll need too many steps to reach a solution
    learning_eps = 1e-8 #Epsilon value for ADAMW - To stabalise the steps taken and avoid large negative ELBO jumps
    num_iter = 1000 #500 #Number of gradient descent steps - needs to run until the ELBO flattens out and the model has converged - we use a <1% change in ELBO over the last 10 itterations to stop
    num_particles = 32 #32 #Number of sample graident calculations at each gradient descent step - power of two values work best
    num_inducing = 1000 #1000 #Number of inducing points to use - number of positions in the conditon grid used to optimise the density distribution. 
    min_iter = 500 #500 #Minimum number of itterations to run before imposing stopping criterion
    stop_prcnt = 0.01 #0.01 #Percentage (fractional) change of the ELBO below which the training stops. We use 1%(0.01) 
    stop_iter = 10 #Number of iterations to look back at to calculate the average ELBO change to impose the stopping criterion
    snapshot_iter = 20 #At each snapshot_iter take a snapshot of the full GP so that it can be loaded and training can begin from their rather than fully restarting if needed
   

    #Prediction set up
    pred_chunk_size = 5000 #Number of prediction grid chunks to be used at one given time to predict on till the full predict grid is filled
    pred_sample_size = 1000 #Number of density/extinction samples to draw from final gp to obtain the percentiles for prediction - We obtain the 16th, 50th, 84th percentiles


    #Rerun statuses if we want to recreate grids or retrain GP from the stars
    recalc_grid_train = True #True/False #If we do/do not want to recalculate the training grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
    recalc_grid_pred = True #True/False #If we do/do not want to recalculate the predicting grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
    recheck_sourcebounds = True #True/False #If we do/do not want to reheck source positions in source data frame to make sure their within our input coorindates and removing any which are not. Also re-precomputes some source indeces for integration optimisation
    retrain_gp = True #True/False  #If we do/do not want to rerun the full GP training based parameters given above - must be recalculated if any of the parameters above change except the prediciting grid parameters
    repredict_gp = True #True/False  #If we do/do not want to rerun the GP prediction based parameters given above - must be recalculated if any of the parameters above change
 
    
    #Run Algorithm on GPUs or CPUs
    train_gpu = True #Set True for GPU run or set False for CPU run for the GP training
    pred_gpu = False #Set True for GPU run or set False for CPU run for the GP predicting. #We also need to set this to False if we're only plotting in a machine with No GPU
    plot_gpu = False  #Set True for plotting in the GPU or set False for CPU for plotting following GPU training especially if plotting is done outside a machine with a GPU
   
    
    ###### End of input parameters which need to be set by user ######


    #Setting if we want to train the full GP from the start or resume training of the GP from a snapshot
    #Will be set as commnad line arguments from NAME script when the run begins
    resume_training = False
    args = sys.argv[1:]
    if "resume_training" in args:
        resume_training = True
        recalc_grid_train = False #We don't want to recalculate a grid everytime the GP training restarts from a previous snapshot
        recalc_grid_pred = False #We don't want to recalculate a grid everytime the GP training restarts from a previous snapshot
        recheck_sourcebounds = False #We don't want to reheck source positions in source data frame to make sure their within our input coorindates and removing any which are not. Also re-precomputes some source indeces for integration optimisation

    #Build/load train grid
    print("Train Grid Calculating")
    l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train = reCalc_TrainGrid(recalc_grid_train, source_df, l_lower_train, l_upper_train, n_l_train, 
                                                                                            b_lower_train, b_upper_train, n_b_train, 
                                                                                            d_min_train, d_max_train, n_d_train)    


    #Checking source positions in source data frame to make sure their within our input coorindates and removing any which are not
    #Also precomputes some source indeces for integration optimisation in the dens_integTorch.py file
    print("Checking if all the sources fall within grid boundaries and removing any which are not")
    source_df, subsample_size, l_inds, b_inds, d_inds = checkSource_bounds(recheck_sourcebounds, source_df, subsample_size, l_bounds_train, b_bounds_train, d_bounds_train)


    #Build/load predict grid
    print("Pred Grid Calculating")
    l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred = reCalc_PredGrid(recalc_grid_pred, source_df, l_lower_pred, l_upper_pred, n_l_pred, 
                                                                                        b_lower_pred, b_upper_pred, n_b_pred, 
                                                                                        d_min_pred, d_max_pred, n_d_pred)
    
    print("Grids calculated and/or loaded")
    # print("l_bounds_train ==", l_bounds_train)
    # print("l_bounds_pred ==", l_bounds_pred)
    # print("b_bounds_train ==", b_bounds_train)
    # print("b_bounds_pred ==", b_bounds_pred)
    print("d_bounds_train ==", d_bounds_train)
    print("d_bounds_pred ==", d_bounds_pred)
    # exit()
    
    

    print("Begin GP stage")

    #Fully train/retrain the GP
    gp, condition_grid = reTrain_GP(retrain_gp, subsample_size, scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                                        learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                                        stop_prcnt, stop_iter, snapshot_iter, resume_training,
                                        l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, source_df, train_gpu, pred_gpu)

    print("GP Model trained and/or loaded")

    
    #Use the GP to predict density and extinction on a chosen Grid
    gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube = rePredict_GP(repredict_gp, pred_chunk_size, pred_sample_size, 
                                                                                                        l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp, pred_gpu, plot_gpu)

    print("GP predicted and/or loaded")

    
   

    #Plot densities and extinctions for analysis
    print("Begin Plotting Ext and Density")

    #Plot training performance plots
    plot_PerformanceMetrics()

    #Plot residuals
    plot_Res_ExtHist(condition_grid, threeDGrid_pred, ext_med_cube, ext_16_cube, ext_84_cube)
    plot_Res_Subtract_ExtHist_NormbyUnc(condition_grid, threeDGrid_pred, ext_med_cube, ext_16_cube, ext_84_cube)

    #Plot final cumilative extinction 
    plot_GP_Pred_ExtCumilative(l_bounds_pred, b_bounds_pred, threeDGrid_pred, ext_med_cube)



    #Plot selected slices along distance of predicted extinction and density 
    plot_GP_Pred_Dens_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gpy_dens_median)
    plot_GP_Pred_Ext_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, ext_med_cube)

    


    
    print("Code Run Time --- %s seconds ---" % (time.time() - start_time))
    
    























