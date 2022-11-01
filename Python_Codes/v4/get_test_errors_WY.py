# Create stream discharge with observational error data (5%, 10%, 25%, 50%, and 100%) -- For all 100 test realizations
# Store as npy 
# AUTHOR: Maruti Kumar Mudunuru

import os
import time
import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
#
from sklearn.metrics import r2_score

np.set_printoptions(precision=2)

#=====================================================;
#  Function-1: Create noisy discharge for obs q-data  ;
#=====================================================;
def get_noisy_obs(r_vec_mat, noise, nq_comps, num_realz, y_obs):

    #--------------------------------------;
    #  Initialize and get noisy obs. data  ;
    #--------------------------------------;
    epsilon   = (1.0/3.0) * noise #Std of the noise value 
    y_obs_mat = np.zeros((num_realz, nq_comps)) #size = (100, 1096)
    #
    for i in range(0,num_realz): #100 realizations
        r_vec            = r_vec_mat[i,:] #Standard normal distribution vector, size = (1096,)
        y_obs_n          = y_obs + epsilon * y_obs * r_vec #Noisy observational discharge, size = (1096,)
        y_obs_n          = np.clip(y_obs_n, a_min = 0.0, a_max = None) #Ensure that we have non-negative discharge
        y_obs_mat[i,:]   = y_obs_n #Create noisy observational data matrix, size = (100, 1096)
        #
        #print(np.min(y_obs), np.min(y_obs_n), np.max(y_obs), np.max(y_obs_n), \
        #    np.mean(y_obs), np.mean(y_obs_n), np.std(y_obs), np.std(y_obs_n))

    return y_obs_mat

#****************************************************************;
#  Set paths, load obs data, and create noisy data (streamflow)  ;
#****************************************************************;
#
if __name__ == '__main__':

    #=========================;
    #  Start processing time  ;
    #=========================;
    tic = time.perf_counter()
    np.random.seed(1337) #For reproducability

    #-------------------------------------------------------;
    #  1. Get test and SWAT params data (all realizations)  ;
    #-------------------------------------------------------;
    path          = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
    path_obs      = path + 'Data/Obs_Data/' #Obs data
    path_swat     = path + 'Data/SWAT/SWAT_v5/' #SWAT data
    path_raw_data = path_swat + 'Raw_Data/' #Raw data
    #
    df_q_obs   = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        		index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016
    x_q_obs    = df_q_obs.values[:,0] #Obs. raw discharge (1096,) -- original data
    nq_comps,  = x_q_obs.shape #No. of discharge points
    #
    num_realz  = 100 #No. of noisy realizations
    realz_list = list(range(1,5*num_realz+1)) #Noisy realization list
    rows_list  = ["R" + str(i) for i in realz_list] #Rows indices for noisy dataframe
    cols_list  = ["t" + str(i) for i in range(0,nq_comps)] #Create pd dataframe header list for q-data
    #
    test_raw_q  = np.load(path_raw_data + "test_raw_q_100.npy") #Raw test data (100, 1096)
    test_raw_p  = np.load(path_raw_data + "test_raw_p_100.npy") #Raw test data (100, 20)

    #-----------------------------------------------------------;
    #  2. Create random vector of 100 realz (for a test realz)  ;
    #-----------------------------------------------------------;
    mu, sigma = 0.0, 1.0 #mean and standard deviation
    r_vec_mat = np.zeros((num_realz, nq_comps)) #size = (100, 1096)
    #
    for i in range(0,num_realz): #100 realizations
        r_vec          = np.random.normal(mu, sigma, nq_comps) #Create standard normal distribution vector, size = 1097
        r_vec_mat[i,:] = copy.deepcopy(r_vec) #Create noisy random data matrix
        #print(i, np.mean(r_vec_mat[i,:]), np.std(r_vec_mat[i,:]))

    #--------------------------------------;
    #  3. Create noisy test q-data for ML  ;
    #     (5%, 10%, 25%, 50%, and 100%)    ;
    #--------------------------------------;
    noise      = 0.05 #Noise value of 5%, 10%, 25%, 50%, and 100%
    noise_list = [0.05, 0.1, 0.25, 0.5, 1.0] #Noise value list
    #
    for noise in noise_list: 
        counter        = 0
        test_q_mat_all = np.zeros((10000,nq_comps)) #(10000,1096)
        #
        for i in range(0,100): #Iterate over each test realz
            test_q_mat                            = get_noisy_obs(r_vec_mat, noise, nq_comps, num_realz, test_raw_q[i,:]) #(100, 1096)
            test_q_mat_all[counter:counter+100,:] = copy.deepcopy(test_q_mat) #(100, 1096)
            counter                               = counter + 100
            for j in range(0,100):
                 print(i,j,r2_score(test_q_mat[j,:],test_raw_q[i,:]))

        df_q_obs_noisy = pd.DataFrame(test_q_mat_all) #Create pd dataframe [10000 rows x 1096 columns]
        np.save(path_swat + "TestErrorData_All/" + "test_raw_q_" + str(noise) + "_noise.npy", test_q_mat_all) #Save error data (10000, 1096) in *.npy file

    #======================;
    # End processing time  ;
    #======================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic)