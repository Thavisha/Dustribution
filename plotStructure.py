import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, AsinhStretch
import cmasher as cmr


"""
Plot GP predicted density and extinction of the regions of interest

"""



#Plot GP predicted density for distance slices
def plot_GP_Pred_Dens_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gpy_dens_median):

    font = {"family":"serif",
            "size":15}
    plt.rc('font', **font)

    image = np.reshape(gpy_dens_median, (n_l_pred, n_b_pred, n_d_pred))

    #We plot a distance slice at every distance boundary we have in the pred grid
    nrows = 3
    ncols = 3
    if len(d_bounds_pred)-1 > nrows*ncols:
        multifig = True
        nfigs = (len(d_bounds_pred)-1)//(nrows*ncols)
        if ((len(d_bounds_pred)-1) % (nrows*ncols)) > 0:
            nfigs+=1

    jslice = 0
    for ifig in range(nfigs): 

        #Dust Density
        fig = plt.figure(figsize=(25, 15)) #width, height

        for islice in range(1,(nrows*ncols)+1):
            slice = jslice+islice - 1

            ax = fig.add_subplot(nrows, ncols, (islice)) #rows, cols

            normalize = ImageNormalize(image, vmin=0.0001, vmax=0.085, stretch=AsinhStretch(a=0.01))
            img = ax.imshow(image[::-1,::-1,slice].T, origin="upper", cmap="cmr.voltage_r", aspect='auto',
                        extent =(np.max(l_bounds_pred), np.min(l_bounds_pred), np.min(b_bounds_pred), np.max(b_bounds_pred)),
                        norm=normalize)

            cbar = plt.colorbar(img, orientation="vertical", shrink=0.7)
            cbar.set_label("Density [mag/pc]")
            ax.set_xlabel("l [deg]")
            ax.set_ylabel("b [deg]")
            plt.title("{:.2f}".format(((d_bounds_pred[slice] + d_bounds_pred[slice+1])/2)) + " pc") 
            if (slice+2) == len(d_bounds_pred):
                break

        jslice+=islice
    
        axes = plt.gcf().get_axes()
        for ax in axes[1:-1:2]:
            ax.set_visible(False)


        plt.tight_layout(pad=1.5, h_pad=0.1, w_pad=0.1)
        plt.savefig("PredDensity_AllSlices_along_Dist"+"_fig"+str(ifig)+".png") 
        #plt.show()
        plt.close()


    return


#Plot GP predicted extinction for distance slices
def plot_GP_Pred_Ext_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, ext_med_cube):

    font = {"family":"serif",
            "size":15}
    plt.rc('font', **font)

    image = ext_med_cube

    #We plot a distance slice at every distance boundary we have in the pred grid
    nrows = 3
    ncols = 3
    if len(d_bounds_pred)-1 > nrows*ncols:
        multifig = True
        nfigs = (len(d_bounds_pred)-1)//(nrows*ncols)
        if ((len(d_bounds_pred)-1) % (nrows*ncols)) > 0:
            nfigs+=1

    jslice = 0
    for ifig in range(nfigs): 

        
        fig = plt.figure(figsize=(25, 15)) #width, height

        for islice in range(1,(nrows*ncols) + 1):
            slice = jslice+islice - 1
            
            ax = fig.add_subplot(3, 3, (islice)) #rows, cols

            normalize = ImageNormalize(image, vmin=0, vmax=np.max(image), stretch=AsinhStretch(a=0.6))
            img = ax.imshow(image[::-1,::-1,slice].T, origin="upper", cmap="cmr.heat_r", aspect='auto',
                        extent =(np.max(l_bounds_pred), np.min(l_bounds_pred), np.min(b_bounds_pred), np.max(b_bounds_pred)),
                        norm=normalize) 

            cbar = plt.colorbar(img, orientation="vertical", shrink=0.7)
            cbar.set_label("Extinction [mag]")
            ax.set_xlabel("l [deg]")
            ax.set_ylabel("b [deg]")
            
            plt.title("{:.2f}".format(((d_bounds_pred[slice] + d_bounds_pred[slice+1])/2)) + " pc") 
            if (slice+2) == len(d_bounds_pred):
                break
            
        jslice+=islice

        axes = plt.gcf().get_axes()
        for ax in axes[1:-1:2]:
            ax.set_visible(False)

        plt.tight_layout(pad=1.2, h_pad=0.1, w_pad=0.1)
        plt.savefig("PredExt_AllSlices_along_Dist"+"_fig"+str(ifig)+".png") 
        #plt.show()
        plt.close()

    return



#Plot cumilative (total) predicted by GP extinction
def plot_GP_Pred_ExtCumilative(l_bounds_pred, b_bounds_pred, threeDGrid_pred, ext_med_cube):


    font = {"family":"serif",
        "size":15}
    plt.rc('font', **font)
    fontsize = 15


    #Cumilative Extinction
    image = ext_med_cube 
    np.save("CumilativeExtinction_Image.pkl", image, allow_pickle=True) #Save Extinction image to compare to Planck Later

    #Plot Cumilative Extinction
    fig = plt.figure(figsize=(15, 11)) #width, height
    ax = fig.add_subplot(1, 1, 1)
    normalize = ImageNormalize(image, vmin=0, vmax=np.max(image), stretch=AsinhStretch(a=0.8))
    img = ax.imshow(image[::-1,::-1,-1].T, origin="upper", aspect="auto", cmap="cmr.heat_r",
                        extent =(np.max(l_bounds_pred), np.min(l_bounds_pred), np.min(b_bounds_pred), np.max(b_bounds_pred)),
                        #vmin=np.min(image), vmax=np.max(image)
                        norm=normalize 
                        )
    cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
    cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
    ax.set_xlabel("l [$^{\\circ}$]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
    ax.set_ylabel("b [$^{\\circ}$]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


    plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
    plt.savefig("Ext_Cumulative.png") 
    #plt.show()
    plt.close()
    return
































   
