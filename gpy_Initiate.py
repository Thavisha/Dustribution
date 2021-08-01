import numpy as np
import torch
import pyro
import tqdm
import time
from memory_profiler import profile
import pickle
from scipy.stats import median_absolute_deviation as mad
from glob import glob
import re
import os

from dens_integTorch import integ_allLoS
from gpyClass_Latent import LatentDensityGPModel





# A quick helper function for getting smoothed percentile values from samples
#Taken from https://docs.gpytorch.ai/en/v1.1.1/examples/07_Pyro_Integration/Cox_Process_Example.html [11]
#Gives the 16th , 50th and 84th percentiles - these will be dens +/- uncertianties in the end
def percentiles_from_samples(samples, percentiles=[0.16, 0.5, 0.84]):
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)
    percentiles_samples = [
        torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentile_samples



@profile
#Train and Condition the GP grid on only the given subsample
#The GP itself must work in Cartesian Coords!
def GP_Train_andCondition(scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                            learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                            stop_prcnt, stop_iter, snapshot_iter, resume_training,
                            l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, condition_grid):



    #Defining the varaible required for Conditioning (eval mode) and Training (in train mode which will use the hp boundaries given in the class)
    #The conditioning data - i.e the value at which the tranformed latent variables (i.e: extinctions) are condictioned on
    condition_coords = torch.tensor(condition_grid[["coords_cartx", "coords_carty", "coords_cartz"]].values, dtype=torch.float) #Define the coordinates for the independent varaible i.e coords
    condition_ext_mean = torch.tensor(condition_grid["A0_p50"].values, dtype=torch.float) #We define the dependent vairiable i.e density mean
    condition_ext_unc = torch.tensor(condition_grid["A0_p50_err"].values, dtype=torch.float) #We define the dependent vairiable i.e density unc (std.dev)

    condition_ext = torch.distributions.Normal(condition_ext_mean, condition_ext_unc).rsample() #We need to provide the density and the dens unc for the training densities


    #Parameters needed for the GP model input
    source_dists = condition_grid["dist_p50"].to_numpy()
    source_l = condition_grid["l"].to_numpy()
    source_b = condition_grid["b"].to_numpy()
    threeDGrid_train_l = threeDGrid_train["pol_l"].to_numpy()
    threeDGrid_train_b = threeDGrid_train["pol_b"].to_numpy()

    l_ind = condition_grid["l_ind"].to_numpy()
    print(l_ind, l_ind.dtype)
    b_ind = condition_grid["b_ind"].to_numpy()
    d_ind = condition_grid["d_ind"].to_numpy()

    #Selecting inducing points
    inducing_point_indeces = torch.randperm(condition_coords.size()[0])[:num_inducing] #Randomly select the number of inducing points from the entire table
    inducing_points = condition_coords[inducing_point_indeces,:]
    torch.set_printoptions(profile="full")
    print("inducing_points before training/learning==", inducing_points)
    torch.set_printoptions(profile="default")
    print("inducing_points size==", inducing_points.size())



    #Create the GP with latent variable
    gp = LatentDensityGPModel(source_dists, source_l, source_b, l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train_l, threeDGrid_train_b, l_ind, b_ind, d_ind,
                                inducing_points, name_prefix="density_gp_model")



    #Defining the set of Hyperparameters to be used in GP (in this case we use scalelength = 2 and mean model = 1 same as we did in Celerite)
    hypers = {
        "mean_module.constant": torch.tensor(mean_ext_dens), #Mean of the GP --> Mean Density in log10(density)
        "covar_module.raw_outputscale": torch.tensor(exp_scalefac), #Scale factor HP for the exp kernel
        "covar_module.base_kernel.raw_lengthscale": torch.tensor([scale_length_x, scale_length_y, scale_length_z]), #For now we use the same scale length for all three exp kernels in the three axe
            }

    #Feeds the Hyperparaters defined above into GP as the initial HPs
    gp.initialize(**hypers) #** => tells python to unpack the dictionary and use each value in the dictionary as an input keyword argument 
    

    #Put GP into train mode for optimization
    gp.train()

    #The coordinates where the latent variables will be infered on - where the GP is trained on to learn the densities (at these given coords)
    train_coords = torch.tensor(threeDGrid_train[["cart_x", "cart_y", "cart_z"]].values, dtype=torch.float, requires_grad = True) #Convert the pandas df to a pytorch compatible data strutcure


    #Empty Arrays to hold iteration information for plotting performance plots later
    elbo_list = []
    lsx_list = []
    lsy_list =[]
    lsz_list = []
    scalefac_list = []
    meanDens_list = []

    # Use the adam+elbo (grad.descent) optimizer in pyro 
    # Here we use AdamW (instead of simple Adam): https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad
    #CLASStorch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)   
    # Here we can train from the start or load a previous snapshot and train from the end of that snapshot - all the information required to restart the gp training from the previous snapshot is saved in the snapshot it self
    def train(elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list, lr=learning_rate, resume_training=resume_training, min_iter = min_iter, num_iter=num_iter, learning_eps=learning_eps, num_particles=num_particles): 

        optimizer = pyro.optim.AdamW({"lr":lr, "eps":learning_eps})
        loss = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=False, retain_graph=True)
        infer = pyro.infer.SVI(gp.model, gp.guide, optimizer, loss=loss)
        loader = tqdm.tqdm(range(num_iter))

        #If we want to resume training from a snapshot it will overwrite the above setup parameters and keep going
        if resume_training:

            #load the most recent Snapshot
            snapshot = load_Snapshot()
            gp.load_state_dict(snapshot["gp_state_dict"])
            optimizer.set_state(snapshot["optimizer_state_dict"])
            loss = snapshot["loss"]
            loader = tqdm.tqdm(range(snapshot["iteration"], num_iter))
            elbo_list = snapshot["elbo_list"]
            lsx_list = snapshot["lsx_list"]
            lsy_list = snapshot["lsy_list"]
            lsz_list = snapshot["lsz_list"]
            scalefac_list = snapshot["scalefac_list"]
            meanDens_list = snapshot["meanDens_list"]
            print("Resuming training from snapshot")
        
        #Regular training loop
        for i in loader:
            loss = infer.step(train_coords, condition_ext_mean, condition_ext_unc)
            loader.set_postfix(loss=loss)
            print("Iter %d/%d - Loss: %.3f   lengthscale_x: %.3f lengthscale_y: %.3f lengthscale_z: %.3f scalefactor: %.3f meanDens: %.3f " % (
                    i + 1, num_iter, loss, #ELBO==LOSS in our case
                    gp.covar_module.base_kernel.raw_lengthscale[:,0].item(),
                    gp.covar_module.base_kernel.raw_lengthscale[:,1].item(),
                    gp.covar_module.base_kernel.raw_lengthscale[:,2].item(),
                    gp.covar_module.raw_outputscale.item(),
                    gp.mean_module.constant.item()
                ))

            #Save iteration information for plotting performance plots later
            elbo_list.append(loss) #ELBO
            lsx_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,0].item()) #Scale length X
            lsy_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,1].item()) #Scale length Y
            lsz_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,2].item()) ##Scale length X
            scalefac_list.append(gp.covar_module.raw_outputscale.item()) #Scale Factor
            meanDens_list.append(gp.mean_module.constant.item()) #Mean Dens

            #Imposing stop criteria - stop after ELBO stop changing by <1% over the last 10 iterations
            #if i > 100: to ensure it"s run for atleast the given minimum number of iterations and doesn"t stop right at the begining.
            #The min number of iterations needs to be larger than the number of iterations averaged over for the stopping criteria
            #To measure the change we use median absolute diviation/median is <1%
            if i > min_iter and (mad(elbo_list[-stop_iter:]))/np.median(elbo_list[-stop_iter:]) < stop_prcnt:
                print("ELBO converged, ending training now")
                snapshot_file = "Snapshot_" + str(i) + ".out" #Snapshot filename - snapshot file saved for the given iteration as a torch file and needs to be reloaded with torc
                save_Snapshot(i, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list) #If the stopping criteria is reached the snapshot is saved at the end of training
                break #Abort training if stopping criteria is reached


            #Making a Snapshot - Checks the current iteration number, finds the remainder compared to the number of iterations afterwhich a snapshot is taken and if the checkpoint is reached it makes a snapshot
            if i % snapshot_iter == 0: #% is the modulo or the remainder operator
                snapshot_file = "Snapshot_" + str(i) + ".out" #Snapshot filename - snapshot file saved for the given iteration as a torch file and needs to be reloaded with torch
                save_Snapshot(i, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list) 

        return elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list

    elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list = train(elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list)

    #Save iteration information for plotting performance plots later
    with open("Iteration_Info.pkl", "wb") as f:
        pickle.dump(elbo_list, f)
        pickle.dump(lsx_list, f)
        pickle.dump(lsy_list, f)
        pickle.dump(lsz_list, f)
        pickle.dump(scalefac_list, f)
        pickle.dump(meanDens_list, f)

    return gp




@profile
#Predict using the previously trained GP model
def GP_Predict(chunk_size, pred_sample_size, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp):

    #Puts the GP into the evaluate mode rather than training mode
    gp.eval()


    #This steps uses the trained gp (trained on only the subsample) to produce the prediction with the input test data sample which in this case is our full grid
    #If we want a new grid for prediction then we need to input it here! 
    pred_coords = torch.tensor(threeDGrid_pred[["cart_x", "cart_y", "cart_z"]].values, dtype=torch.float, requires_grad = True) #Convert the pandas df to a pytorch compatible data strutcure

    gp_DensPred_start_time = time.time()
    print("GP Dens Pred start time = ", gp_DensPred_start_time)

    #We loop over chunks of the predicting grid to make sure there is no memory issues
    #This way we can even predict on signle coord points at a time
    with torch.no_grad(): #Removes the grad info since we don"t need it - our latent function now is the density so we don"t need grads
        
        i_start = 0
        i_end = i_start + chunk_size
        print("pred_coords size==", pred_coords.size())

        while i_start < pred_coords.size()[0]:

            try:
                function_dist = gp(pred_coords[i_start:i_end,:])  #Take a distribution of the latent varaibles/samples
                dens_samples = 10**function_dist(torch.Size([pred_sample_size])) #sample from that distribution and transform to the function domain we want - in our case log10(dens) to dens
                lowerP, median, upperP = percentiles_from_samples(dens_samples) #Gives 16th, 50th, 84th percentiles
            except ValueError:
                function_dist = gp(pred_coords[i_start:,:])  
                dens_samples = 10**function_dist(torch.Size([pred_sample_size])) 
                lowerP, median, upperP = percentiles_from_samples(dens_samples) 

            try:
                dens_samples_all = torch.cat((dens_samples_all, dens_samples), 1) 
                gpy_dens_median = torch.cat((gpy_dens_median, median), 0) #Median (50th Percentile)
                gpy_dens_16P = torch.cat((gpy_dens_16P, lowerP), 0) #16th Percentile
                gpy_dens_84P = torch.cat((gpy_dens_84P, upperP), 0) #84th Percentile
            except UnboundLocalError:
                dens_samples_all = dens_samples
                gpy_dens_median = median
                gpy_dens_16P = lowerP
                gpy_dens_84P = upperP

            i_start = i_end
            i_end += chunk_size

            print("dens_samples_all size",dens_samples_all.size())

    print("GP Dens Predict Run Time --- %s seconds ---" % (time.time() - gp_DensPred_start_time))
    print("Dens Pred completed Begin Ext Integration")


    gp_ExtPred_start_time = time.time()
    print("GP Ext Pred start time = ", gp_ExtPred_start_time)

    #Integrate all the density samples to get Extinctions percentiles
    extinction_hypercube = integ_allLoS(l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred["pol_l"].to_numpy(), threeDGrid_pred["pol_b"].to_numpy(), dens_samples_all.numpy(), n_samples=pred_sample_size)

    ext_med_cube = np.percentile(extinction_hypercube, 50, axis = 0)
    ext_16_cube = np.percentile(extinction_hypercube, 16, axis = 0)
    ext_84_cube = np.percentile(extinction_hypercube, 84, axis = 0)

    print("GP Ext Predict Run Time --- %s seconds ---" % (time.time() - gp_ExtPred_start_time))
    print("Extinction Integration Completed")

    return gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube


#Function to Load a Saved GP Model
def load_GPmodel(num_inducing, condition_grid, threeDGrid_train, l_bounds_train, b_bounds_train, d_bounds_train, gp_filename):

    gp_dict = torch.load(gp_filename)

    source_dists = condition_grid["dist_p50"].to_numpy()
    source_l = condition_grid["l"].to_numpy()
    source_b = condition_grid["b"].to_numpy()
    condition_coords = torch.tensor(condition_grid[["coords_cartx", "coords_carty", "coords_cartz"]].values, dtype = torch.float) 
    condition_ext_mean = torch.tensor(condition_grid["A0_p50"].values, dtype = torch.float) 
    condition_ext_unc = torch.tensor(condition_grid["A0_p50_err"].values, dtype = torch.float)
    
    threeDGrid_train_l = threeDGrid_train["pol_l"].to_numpy()
    threeDGrid_train_b = threeDGrid_train["pol_b"].to_numpy()
    inducing_points = condition_coords[:num_inducing,:]

    l_ind = condition_grid["l_ind"].to_numpy()
    b_ind = condition_grid["b_ind"].to_numpy()
    d_ind = condition_grid["d_ind"].to_numpy()

    
    gp = LatentDensityGPModel(source_dists, source_l, source_b, l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train_l, threeDGrid_train_b, 
                                l_ind, b_ind, d_ind, inducing_points, name_prefix="density_gp_model")
    gp.load_state_dict(gp_dict)
    
    return gp


#Saving snapshot at the given checkpoint 
def save_Snapshot(iteration, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list):
    
    #Create and save the most recent snapshot
    torch.save({"iteration":iteration, "gp_state_dict":gp.state_dict(), "optimizer_state_dict": optimizer.get_state(), "loss":loss, "elbo_list":elbo_list, "lsx_list":lsx_list, "lsy_list":lsy_list, "lsz_list":lsz_list, "scalefac_list":scalefac_list, "meanDens_list":meanDens_list}, 
                    snapshot_file)


    # #Delete snapshots older than the last five saved (we can save more if we want to)
    # snapshots = glob("Snapshot_*.out") #Pickout all snapshot files to figure out which iteration they are from - gives a list of the snapshot file names

    # number_of_snapshots_tosave = 2 #Delete all snapshots except the five highest (last) iterations

    # if len(snapshots) > number_of_snapshots_tosave:
    #     snapshot_iters = np.array([ np.int(re.split("[_.]", snapshot)[1]) for snapshot in snapshots])
    #     snapshot_list_ascend = list(np.array(snapshots)[np.argsort(snapshot_iters)]) #Arrange the snapshot list by asending iteration number
    #     for f in snapshot_list_ascend[:-number_of_snapshots_tosave]:
    #         os.remove(f) #Delete all snapshots except the five highest (last) iterations


    return




#Loading snapshot at the given checkpoint 
def load_Snapshot():

    #Pickout all snapshot files to figure out which iteration they are from - gives a list of the snapshot file names
    snapshots = glob("Snapshot_*.out")

    #Identify the most recent snapshot in the directory which is the highest iteration numbered snapshot
    snapshot_iters = [int(re.split("[_.]", snapshot)[1]) for snapshot in snapshots] #split the file name to idenitfy the snapshot numbers to get the iteration number
    most_recent_file = snapshots[np.argmax(snapshot_iters)]
    
    return torch.load(most_recent_file) #, np.argmax(snapshot_iters), most_recent_file












