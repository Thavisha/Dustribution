import numpy as np
import pandas as pd
from astropy.coordinates import spherical_to_cartesian



"""
Run the GPytorch on a spherical polar coords grid

Gpytorch doesn't care if the grid is in sperical polar coords or cartesian coords since it's in 3D

So to speed up our integration of the LoS we can simply use a spherical polar grid and only fit the GP in cart coords

We place the sun in the middle (0,0,0) in both sets of coords 

"""


#A spherical polar coords grid which also knows about its correposing cartesian coords
def threeD_grid(l_lower, l_upper, n_l, b_lower, b_upper, n_b, d_min, d_max, n_d): 



    #Gives boundaries of l,b,d grid cells
    l_bounds = np.linspace(l_lower, l_upper, n_l+1) #+1: Need one more bound than the number of cells to get the last bound
    b_bounds = np.rad2deg(np.arcsin(np.linspace(np.sin(np.deg2rad(b_lower)), np.sin(np.deg2rad(b_upper)), n_b+1)))
    #d_bounds = np.logspace(np.log10(d_min), np.log10(d_max), n_d+1) #dist in pc and logspacing
    d_bounds = np.linspace(d_min, d_max, n_d+1) #dist in pc and linear spacing

    rows = []


    for i in range(len(l_bounds)-1): #-1 to remove the last bound element so that we don't do i+1 on last element since there is no +1
        for j in range(len(b_bounds)-1):
            for k in range(len(d_bounds)-1):

                #Calc mid points of grid cells for the spherical polar coords
                l_mid = (l_bounds[i] + l_bounds[i+1])/2
                b_mid = (b_bounds[j] + b_bounds[j+1])/2
                #d_mid = 10**( ( np.log10(d_bounds[k]) + np.log10(d_bounds[k+1]) )/2 ) #log spacing distance
                d_mid = (d_bounds[k] + d_bounds[k+1])/2 #linear spaced distance
                

                #Calc mid points in Cart coords - required to place blob and fill density
                #Need to give the input as dist, lat, long (d,b,l)
                x_mid, y_mid, z_mid = spherical_to_cartesian(d_mid, np.deg2rad(b_mid), np.deg2rad(l_mid)) 


                rows.append([l_mid, b_mid, d_mid, x_mid, y_mid, z_mid]) 

    

    threeDGrid = pd.DataFrame(rows, columns=["pol_l", "pol_b", "pol_d", "cart_x", "cart_y", "cart_z"], dtype="float")


    return l_bounds, b_bounds, d_bounds, threeDGrid 




























