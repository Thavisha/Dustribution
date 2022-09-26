import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


"""
Plot residuals of input data compared to GP predicted data

e.g: residual of input extinctions - gp model predicted extinctions

"""


#Subtracted residual histogram of extinctions (input extinctions - predicted model extinctions)
def plot_Res_ExtHist(condition_grid, threeDGrid, ext_med_cube, ext_16_cube, ext_84_cube): 

    #Load input extinction
    ext_original = condition_grid["A0_p50"].to_numpy()

    fig = plt.figure(figsize=(10, 10)) #width, height

    #Input extinction histogram
    n, bins, _patches = plt.hist(ext_original, bins=50, color="purple", histtype="step", linewidth=5, label="Input extinction", zorder=3)


    #GP predicted extinction at the position of the stars - need to interpolation for this
    interp_function = RegularGridInterpolator((
                                                np.unique(threeDGrid["pol_l"].to_numpy()), 
                                                np.unique(threeDGrid["pol_b"].to_numpy()), 
                                                np.unique(threeDGrid["pol_d"].to_numpy())#[:24]
                                                ), 
                                                ext_med_cube[:,:,1:], method='linear', bounds_error=False, fill_value=None)

    interp_points = np.array([[condition_grid["l"].to_numpy()[i],condition_grid["b"].to_numpy()[i], 
                                condition_grid["dist_p50"].to_numpy()[i]] for i in range(len(condition_grid["dist_p50"].to_numpy()))])
    interp_pred_ext = interp_function(interp_points)

    plt.hist(interp_pred_ext, bins=bins, color="orange", histtype="step", linewidth=5, label="Predicted extinction", zorder=1)

    plt.xlabel("Extinction")
    plt.ylabel("Number of sources")
    plt.legend(fontsize=14, loc="upper right")
    plt.savefig("Extinction_Input_andPred_Hist.png")
    #plt.show()
    plt.close()


    #Subtracted Residual: Predicted extinction - Input extinction
    res_sub = interp_pred_ext - ext_original
    fig = plt.figure(figsize=(10, 10)) #width, height
    plt.hist(res_sub, bins=50, color="black", histtype="step", linewidth=5)
    plt.xlabel("Extinction residual (predicted - input)")
    plt.ylabel("Number of sources")
    plt.title("Extinction residual (predicted - input)")
    plt.savefig("Extinction_Input_minPred_ResHist.png")
    #plt.show()
    plt.close()

    #Ratio Residual Predicted extinctio/Input extinction
    res_ratio = interp_pred_ext/ext_original
    fig = plt.figure(figsize=(10, 10)) #width, height
    plt.hist(res_ratio, bins=50, color="pink", histtype="step", linewidth=5)
    plt.xlabel("Extinction residual")
    plt.ylabel("Number of sources")
    plt.title("Extinction residual (predicted/input)")
    plt.savefig("Extinction_Input_divPred_ResHist.png")
    #plt.show()
    plt.close()

    #Ratio Residual Input extinction/Predicted extinction
    res_ratio = ext_original/interp_pred_ext
    fig = plt.figure(figsize=(10, 10)) #width, height
    plt.hist(res_ratio, bins=500, color="green", histtype="step", linewidth=5)
    plt.xlabel("Extinction A0 Residual")
    plt.ylabel("Number of Sources")
    plt.title("Extinction residual (input/predicted)")
    plt.savefig("Extinction_Pred_divInput_ResHist.png")
    #plt.show()
    plt.close()




#Subtracted Residual Extinction Histogram of only Exts (Original - Predicted) normalised by extinction uncertianties
def plot_Res_Subtract_ExtHist_NormbyUnc(condition_grid, threeDGrid, ext_med_cube, ext_16_cube, ext_84_cube): 

    ext_original = condition_grid["A0_p50"].to_numpy()
    ext_original_unc = condition_grid["A0_p50_err"].to_numpy()
    

    #Predicted Extinction at the Position of the stars - Need to interpolation for this
    interp_function = RegularGridInterpolator((
                                                np.unique(threeDGrid["pol_l"].to_numpy()), 
                                                np.unique(threeDGrid["pol_b"].to_numpy()), 
                                                np.unique(threeDGrid["pol_d"].to_numpy())#[:24]
                                                ), 
                                                ext_med_cube[:,:,1:], method='linear', bounds_error=False, fill_value=None)

    interp_points = np.array([[condition_grid["l"].to_numpy()[i],condition_grid["b"].to_numpy()[i], 
                                condition_grid["dist_p50"].to_numpy()[i]] for i in range(len(condition_grid["dist_p50"].to_numpy()))])
    interp_pred_ext = interp_function(interp_points)
                                   
    interp_function_16P = RegularGridInterpolator((
                                                np.unique(threeDGrid["pol_l"].to_numpy()), 
                                                np.unique(threeDGrid["pol_b"].to_numpy()), 
                                                np.unique(threeDGrid["pol_d"].to_numpy())#[:24]
                                                ), 
                                                ext_16_cube[:,:,1:], method='linear', bounds_error=False, fill_value=None)
    interp_pred_ext_16P = interp_function_16P(interp_points)

    interp_function_84P = RegularGridInterpolator((
                                                np.unique(threeDGrid["pol_l"].to_numpy()), 
                                                np.unique(threeDGrid["pol_b"].to_numpy()), 
                                                np.unique(threeDGrid["pol_d"].to_numpy())#[:24]
                                                ), 
                                                ext_84_cube[:,:,1:], method='linear', bounds_error=False, fill_value=None)
    interp_pred_ext_84P = interp_function_16P(interp_points)
    
    interp_pred_ext_unc = ((interp_pred_ext_84P - interp_pred_ext) + (interp_pred_ext - interp_pred_ext_16P))/2

    combined_unc = np.sqrt((ext_original_unc**2) + (interp_pred_ext_unc**2))

    #Subtracted Residual
    res_sub = interp_pred_ext - ext_original
    norm_res = res_sub / combined_unc
    fig = plt.figure(figsize=(10, 10)) #width, height
    plt.hist(norm_res, bins=50, color="grey", histtype="step", linewidth=5)
    plt.xlabel("Residual")
    plt.ylabel("Number of sources")
    plt.title("Extinction residual normalised by combined uncertianties" + "\n" + " (Pred-Original / combinedUnc)")
    plt.savefig("Extinction_Norm_byUnc_Res_Hist.png")
    #plt.show()
    plt.close()







    