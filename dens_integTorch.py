import numpy as np
import torch


"""
Integrate along (grid/source) LoS to derive integrated dust density from the density maps derived by gpytorch
The integrated dust densities from the grid will be the model extinctions which we can compare to out observed data

"""

#Integrate density for ALL stars tp Obtain the (integrated) model extinctions - Units of Ext = Mags
def integ_allSource(source_dists, source_l, source_b, l_bounds, b_bounds, d_bounds, threeDGrid_l, threeDGrid_b, l_ind, b_ind, d_ind, density_samples):
    
    """
    this returns the whole distribution of pathInts
    """

    #Empty array to hold path integrated density distributions in the end
    #If we have multiple dust types then we just add more sets of zero arrays as cols
    PredExt_Distribution = torch.zeros(len(source_dists))

    #global source_indices
    source_indices = torch.arange(len(source_dists)) #hack to get around need to know indexes in loop over sources - this could be skipped if an input was created as source_indices with source_df.index.values passed to that argument

    #Calculate the integral at the outer edge of every cell.
    #we concatenate a set of zeros first because the integral at the inner edge of the grid must be 0
    #Then the cumulative sum of the line integral of each cell is concatenated to that, as that is our approximation of the integral at the outer edge of the cell
    integral_grid = torch.cat((torch.zeros((len(l_bounds)-1,len(b_bounds)-1,1), requires_grad=True), 
                              torch.cumsum(density_samples.reshape(len(l_bounds)-1,len(b_bounds)-1,len(d_bounds)-1) * (d_bounds[1:] - d_bounds[:-1]), dim=2)), dim=2)
    
    
    PredExt_Distribution =  integral_grid[l_ind,b_ind,d_ind] + ((integral_grid[l_ind,b_ind,d_ind+1] - integral_grid[l_ind,b_ind,d_ind])*(source_dists - d_bounds[d_ind]))



    
    #pathInt_Distribution_Dist here is UNITLESS, we need to add the units into it in the calc_Likelihood stage
    return PredExt_Distribution





#Integrate over every LoS in Grid (independent of source positions) in order to return the final extinction map and dust column densities 
def integ_allLoS(l_bounds, b_bounds, d_bounds, threeDGrid_l, threeDGrid_b, dens_samples_all, n_samples=1):
    ##### RIGHT NOW THIS COMPUTES TO THE OUTER EDGE OF THE CELL. RETURNS N_D INTEGRALS TO THE OUTER EDGES, WITH A SET OF ZEROS AT THE INNERMOST EDGE OF THE GRID ####

    col_densities = np.concatenate((np.zeros((n_samples,len(l_bounds)-1,len(b_bounds)-1,1)), 
                              np.cumsum(dens_samples_all.reshape(n_samples,len(l_bounds)-1,len(b_bounds)-1,len(d_bounds)-1) * (d_bounds[1:] - d_bounds[:-1]), axis=3)), axis=3)

    return col_densities



















