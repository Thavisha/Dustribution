import numpy as np
import pandas as pd
import time
import sys
import torch
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian

from threeD_Grid_Build import threeD_grid
from gpy_Initiate import GP_Train_andCondition, GP_Predict, load_GPmodel
from plotStructure import plot_GP_Pred_Dens_Slices_AlongDist, plot_GP_Pred_Ext_Slices_AlongDist, plot_GP_Pred_ExtCumilative
from plotResiduals import plot_Res_ExtHist, plot_Res_Subtract_ExtHist_NormbyUnc
from plotPerformance import plot_PerformanceMetrics


#Read in the input table and build a source pandas data frame
def read_inputTable(input_filename):


    #Read in input data from source table
    source_df_tab = pd.read_csv(input_filename)
    source_df = source_df_tab[["l", "b", "dist_p50",  "A0_p50", "A0_p16", "A0_p84"]]


    #Calculate extinction uncertianties from the input exticntion percentiles 
    source_df.loc[:, ("A0_p50_err")] = ( (source_df["A0_p84"] - source_df["A0_p50"]) + (source_df["A0_p50"] - source_df["A0_p16"]) ) / 2
    

    #Create a set of cartesian coordinates xyz from the input spherical polar cooridnates lbd to be used in the GP
    coords_x, coords_y, coords_z = spherical_to_cartesian(source_df["dist_p50"].to_numpy(), np.deg2rad(source_df["b"].to_numpy()), np.deg2rad(source_df["l"].to_numpy()))
    source_df.loc[:, ("coords_cartx")] = coords_x 
    source_df.loc[:, ("coords_carty")] = coords_y
    source_df.loc[:, ("coords_cartz")] = coords_z 

    
    #Mask out NaN extinction values
    source_df = source_df[source_df["A0_p50"].notna()]  


    # #Display input table of parameters if required
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(source_df)
    # exit()

    return source_df


#Build/load train grid
def reCalc_TrainGrid(recalc_grid_train, source_df, l_lower_train, l_upper_train, n_l_train, 
                        b_lower_train, b_upper_train, n_b_train, 
                        d_min_train, d_max_train, n_d_train):
    
    if recalc_grid_train:
        
        #Build the 3D Train grid 
        l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train =  threeD_grid(l_lower_train, l_upper_train, n_l_train, b_lower_train, b_upper_train, n_b_train, 
                                                                                            d_min_train, d_max_train, n_d_train)

        #Pickle train grid for storing
        threeDGrid_train.to_pickle("threeDGrid_train.pkl")
        np.save("l_bounds_train.pkl",l_bounds_train, allow_pickle=True)
        np.save("b_bounds_train.pkl",b_bounds_train, allow_pickle=True)
        np.save("d_bounds_train.pkl",d_bounds_train, allow_pickle=True)

    else: 
        #Load train grid from pickled data
        print("Loading Train Grid")
        l_bounds_train = np.load("l_bounds_train.pkl.npy", allow_pickle=True)
        b_bounds_train = np.load("b_bounds_train.pkl.npy", allow_pickle=True)
        d_bounds_train = np.load("d_bounds_train.pkl.npy", allow_pickle=True)
        threeDGrid_train = pd.read_pickle("threeDGrid_train.pkl")
    # exit()

    return l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train


#Checking source positions in source data frame to make sure their within our input coorindates and removing any which are not
#Also precomputes some source indeces for integration optimisation in the dens_integTorch.py file   
def checkSource_bounds(recheck_sourcebounds, source_df, subsample_size, l_bounds_train, b_bounds_train, d_bounds_train):   

    if recheck_sourcebounds:

        #We need to identify which cell each source is in
        l_inds = []
        b_inds = []
        d_inds = []

        for i in range(len(source_df)):
            try:
                l_inds.append(np.argwhere(l_bounds_train < source_df.loc[i,'l'])[-1][0])
            except IndexError: #Above line triggers IndexError if l < l_min
                l_inds.append(-1)
            try:    
                b_inds.append(np.argwhere(b_bounds_train < source_df.loc[i,'b'])[-1][0])
            except IndexError:
                b_inds.append(-1)
            try:
                d_inds.append(np.argwhere(d_bounds_train < source_df.loc[i,'dist_p50'])[-1][0])
            except IndexError:
                d_inds.append(-1)
        
        source_df['l_ind'] = l_inds
        source_df['b_ind'] = b_inds
        source_df['d_ind'] = d_inds

        # print(source_df)
        # print(len(l_bounds_train), len(b_bounds_train), len(d_bounds_train))
        # print(np.max(l_inds), np.max(b_inds), np.max(d_inds))

        l_inds = np.array(l_inds)
        b_inds = np.array(b_inds)
        d_inds = np.array(d_inds)
        outgrid=np.logical_or(
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(l_inds < 0, l_inds >= len(l_bounds_train)-1), 
                                    b_inds < 0),
                            b_inds >= len(b_bounds_train)-1),
                        d_inds < 0),
                d_inds >= len(d_bounds_train)-1)
        # print(np.sum(outgrid))

        ingrid = np.logical_not(outgrid)
        source_df = source_df[ingrid] #Exclude sources which are actually outside the train grid that has been defined
    
        np.save("l_inds.pkl",l_inds, allow_pickle=True)
        np.save("b_inds.pkl",b_inds, allow_pickle=True)
        np.save("d_inds.pkl",d_inds, allow_pickle=True)
        source_df.to_pickle("sources_InBounds.pkl")

   
    else:

        #Load indeces saved for intergration later as well as the source table for the sources within the given bounds
        l_inds = np.load("l_inds.pkl.npy", allow_pickle=True)
        b_inds = np.load("b_inds.pkl.npy", allow_pickle=True)
        d_inds = np.load("d_inds.pkl.npy", allow_pickle=True)
        source_df = pd.read_pickle("sources_InBounds.pkl")


    #Checks if subsample size and source_df are sensible numbers and not zero!
    if subsample_size > len(source_df): #We have now removed a few items from the table, so we need to make sure we still only ask for no more rows than the number that exist
        subsample_size = len(source_df)
    # exit()
    if len(source_df) < 1:
            raise RuntimeError("The number of input sources is less than one. Please check the input table and train grid & predict grid boundaries carefully!")

    return source_df, subsample_size, l_inds, b_inds, d_inds


#Build/load predict grid
def reCalc_PredGrid(recalc_grid_pred, source_df, l_lower_pred, l_upper_pred, n_l_pred, 
                        b_lower_pred, b_upper_pred, n_b_pred, 
                        d_min_pred, d_max_pred, n_d_pred):

    if recalc_grid_pred:
        
        print("Predict Grid Calculating")

        #Build the 3D predict grid 
        l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred =  threeD_grid(l_lower_pred, l_upper_pred, n_l_pred, b_lower_pred, b_upper_pred, n_b_pred, 
                                                                                        d_min_pred, d_max_pred, n_d_pred)

        threeDGrid_pred.to_pickle("threeDGrid_pred.pkl")
        np.save("l_bounds_pred.pkl",l_bounds_pred, allow_pickle=True)
        np.save("b_bounds_pred.pkl",b_bounds_pred, allow_pickle=True)
        np.save("d_bounds_pred.pkl",d_bounds_pred, allow_pickle=True)

    else:
        #Reload predict grid from pickled data
        print("Loading Predict Grid")
        l_bounds_pred = np.load("l_bounds_pred.pkl.npy", allow_pickle=True)
        b_bounds_pred = np.load("b_bounds_pred.pkl.npy", allow_pickle=True)
        d_bounds_pred = np.load("d_bounds_pred.pkl.npy", allow_pickle=True)
        threeDGrid_pred = pd.read_pickle("threeDGrid_pred.pkl")


    return l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred


#Fully train/retrain the GP
def reTrain_GP(retrain_gp, subsample_size, scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                stop_prcnt, stop_iter, snapshot_iter, resume_training,
                l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, source_df, train_gpu, pred_gpu):

    if retrain_gp:

        print("Training GP")

        #Define subsample of sources on which the gp is to be conditioned on - if subsample_size is set to full input source_df length then we use all the sources from the catalog that meet our requirements like within conditon grid and not a NAN
        condition_grid = source_df.sample(n=subsample_size, random_state=977) 
        
        #Final source sample used to condition the GP
        condition_grid.to_pickle("condition_grid.pkl")
        print("Final source sample used as input for GP ==", len(condition_grid))
        if len(condition_grid) < 1:
            raise RuntimeError("The number of input sources is less than one. Please check the input table and train grid & predict grid boundaries carefully!")

        gp_start_time = time.time()
        print("GP train start time = ", gp_start_time)
        
        #Run the GP  
        gp = GP_Train_andCondition(scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                                    learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                                    stop_prcnt, stop_iter, snapshot_iter, resume_training,
                                    l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, condition_grid, train_gpu)
        print("GP Train Run Time --- %s seconds ---" % (time.time() - gp_start_time))

        #Save the GP for prediciting 
        torch.save(gp.state_dict(), "gp_trained.out")

    
    else: 

        print("Loading trained GP")
        
        #If we don't want to retrain the GP simply load the pre-trained GP 
        condition_grid = pd.read_pickle("condition_grid.pkl")
        print("Final source sample used as input for GP ==", len(condition_grid))
        if len(condition_grid) < 1:
            raise RuntimeError("The number of input sources is less than one. Please check the input table and train grid & predict grid boundaries carefully!")
        
        #Correctly load the GP model 
        gp = load_GPmodel(num_inducing, condition_grid, threeDGrid_train, l_bounds_train, b_bounds_train, d_bounds_train, train_gpu, pred_gpu, gp_filename="gp_trained.out")
        # exit()

    return gp, condition_grid



#Use the GP to predict density and extinction on a chosen Grid
def rePredict_GP(repredict_gp, pred_chunk_size, pred_sample_size, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp, pred_gpu, plot_gpu):

    if repredict_gp:

        print("GP Prediciting")

        #Predict the density and extinction with trained GP model
        gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube = GP_Predict(pred_chunk_size, pred_sample_size, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp, pred_gpu)

        torch.save(gpy_dens_median, "gpy_dens_median.out")
        torch.save(gpy_dens_16P, "gpy_dens_16P.out")
        torch.save(gpy_dens_84P, "gpy_dens_84P.out")
        np.save("ext_med_cube.pkl", ext_med_cube, allow_pickle=True)
        np.save("ext_16_cube.pkl", ext_16_cube, allow_pickle=True)
        np.save("ext_84_cube.pkl", ext_84_cube, allow_pickle=True)


    else:

        #If we don't want to repredict the GP simply load the pre-predicted density and extinctions for analysis 
        print("Loading GP Predicitions")
        
        if plot_gpu: #Need to set this here so that if we're plotting outside the GPU then the file is loaded properly

            gpy_dens_median = torch.load("gpy_dens_median.out")
            gpy_dens_16P = torch.load("gpy_dens_16P.out")
            gpy_dens_84P = torch.load("gpy_dens_84P.out")

        else: 
            #Need this here to open the file properly if we have trained and predicted in a GPU but want to do the plotting in a CPU
            gpy_dens_median = torch.load("gpy_dens_median.out", map_location=torch.device("cpu"))
            gpy_dens_16P = torch.load("gpy_dens_16P.out", map_location=torch.device("cpu"))
            gpy_dens_84P = torch.load("gpy_dens_84P.out", map_location=torch.device("cpu"))        

        ext_med_cube = np.load("ext_med_cube.pkl.npy", allow_pickle=True)
        ext_16_cube = np.load("ext_16_cube.pkl.npy", allow_pickle=True)
        ext_84_cube = np.load("ext_84_cube.pkl.npy", allow_pickle=True)
        #gpy_mean.eval()


    #Convert from torch tensor to numpy array for plotting
    gpy_dens_median = gpy_dens_median.cpu().detach().numpy() 
    gpy_dens_16P = gpy_dens_16P.cpu().detach().numpy()
    gpy_dens_84P = gpy_dens_84P.cpu().detach().numpy()


    return gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube













