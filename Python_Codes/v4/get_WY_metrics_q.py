# Get SWAT ensemble performance with respect to Obs data (SWAT_v5 -- 1000 sobol sequence with only one soil layer)
# Metrics --> R2-score, MSE, RMSE, NSE, logNSE, KGE, MPE--MeanPercentError (VolumeError)
#   Water Year (WY) from 2014 to 2016
#   Oct-1-2013 to Sep-30-2016
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import pickle
import time
import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hydroeval as he  #An evaluator for streamflow time series in Python
#
import sklearn #'1.0.2'
from sklearn.metrics import mean_squared_error, r2_score

np.set_printoptions(precision=2)
print("sklearn version = ", sklearn.__version__)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path           = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat      = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs       = path + 'Data/Obs_Data/' #Obs data
path_ind       = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
path_pp_models = path + 'PreProcess_Models/' #Pre-processing models for standardization
path_pp_data   = path_swat + 'PreProcessed_Data/' #Pre-processed data
path_raw_data  = path_swat + 'Raw_Data/' #Raw data

#************************************;
#  2a. SWAT data for pre-processing  ;
#************************************;
df_p           = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
                        		index_col = 0) #[1000 rows x 20 columns]
df_q           = pd.read_csv(path_swat + 'flow_wy_sobol_1000realz.csv', \
                        		index_col = 0) #[1096 rows x 1001 columns] WY-2014 to WY-2016
df_q_obs       = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        		index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016
#
#['CN2', 'ESCO', 'RCHRG_DP', 'GWQMN', 'GW_REVAP', 'REVAPMN', 'GW_DELAY', 'ALPHA_BF', \
# 'SOL_K', 'SOL_AWC', 'CH_N2', 'CH_K2', 'OV_N', 'SFTMP', 'SMTMP',
# 'SMFMX', 'TIMP', 'EPCO', 'PLAPS', 'TLAPS'] #Total number of 20 parameters
#
swat_p_list    = df_p.columns.to_list() #SWAT params list -- A total of 20; length = 20
q_list         = df_q.columns.to_list()[1:] #Discharge realz list -- A total of 1000; length = 1000
#
x_p            = df_p.values #Matrix of values for SWAT params -- 1000 realz; (1000, 20)
x_q            = np.transpose(df_q.iloc[:,1:].values) #Matrix of values of discharge -- 1000 realz; (1000, 1096)
#
x_q_obs        = df_q_obs.values[:,0] #Obs. raw discharge (1096,)
#
#mi_ind_list    = [14, 12, 8, 3, 11, 15, 7, 0, 16] #Order of sensitivities, a total of 9
#swat_p_mi_list = [swat_p_list[i] for i in mi_ind_list] #SWAT sensitive parameters a total of 9

#*********************************************;
#  3. SWAT q vs obs metrics (train/val/test)  ;
#*********************************************;
num_realz    = 1000 #Sobol realz (total realz data)
num_train    = 800 #Training realz
num_val      = 100 #Validation realz
num_test     = 100 #Testing realz
#
metrics_list = np.zeros((num_realz,10), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_list)))
#
for i in range(0,num_realz):
    r2          = r2_score(x_q_obs, x_q[i,:]) #r2-score
    mse         = mean_squared_error(x_q_obs, x_q[i,:]) #MSE-score
    rmse        = mean_squared_error(x_q_obs, x_q[i,:], squared = False) #RMSE-score
    nse         = he.evaluator(he.nse, x_q_obs, x_q[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs), np.log10(x_q[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs, x_q[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs, x_q[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs, x_q[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    mpe_ve      = 100*(np.mean(x_q[i,:]) - np.mean(x_q_obs))/np.mean(x_q_obs) #Mean percent error or volume error
    print(i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve))
    #
    metrics_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
                        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve)

cols_list  = ['i', 'r2', 'mse', 'rmse', 'nse', 'lognse', 'kge', 'mkge', 'npkge', 'mpe_ve']
df_metrics = pd.DataFrame(metrics_list, columns = cols_list)
#
print('r2-score min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_list[:,1])), '{0:.2g}'.format(np.mean(metrics_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_list[:,1]))) #r2-score min, max, mean, std =  -6.4 0.57 -1.2 1.2
#
print('mse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,2])), \
        '{0:.2g}'.format(np.max(metrics_list[:,2])), '{0:.2g}'.format(np.mean(metrics_list[:,2])), \
        '{0:.2g}'.format(np.std(metrics_list[:,2]))) #mse min, max, mean, std =  21 3.6e+02 1.1e+02 61
#
print('rmse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,3])), \
        '{0:.2g}'.format(np.max(metrics_list[:,3])), '{0:.2g}'.format(np.mean(metrics_list[:,3])), \
        '{0:.2g}'.format(np.std(metrics_list[:,3]))) #rmse min, max, mean, std =  4.6 19 10 2.9
#
print('nse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,4])), \
        '{0:.2g}'.format(np.max(metrics_list[:,4])), '{0:.2g}'.format(np.mean(metrics_list[:,4])), \
        '{0:.2g}'.format(np.std(metrics_list[:,4]))) #nse min, max, mean, std =  -14 0.62 0.046 0.91
#
print('lognse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,5])), \
        '{0:.2g}'.format(np.max(metrics_list[:,5])), '{0:.2g}'.format(np.mean(metrics_list[:,5])), \
        '{0:.2g}'.format(np.std(metrics_list[:,5]))) #lognse min, max, mean, std =  -35 0.72 0.066 1.8
#
print('kge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,6])), \
        '{0:.2g}'.format(np.max(metrics_list[:,6])), '{0:.2g}'.format(np.mean(metrics_list[:,6])), \
        '{0:.2g}'.format(np.std(metrics_list[:,6]))) #kge min, max, mean, std =  -2.2 0.74 0.32 0.23
#
print('mkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,7])), \
        '{0:.2g}'.format(np.max(metrics_list[:,7])), '{0:.2g}'.format(np.mean(metrics_list[:,7])), \
        '{0:.2g}'.format(np.std(metrics_list[:,7]))) #mpe_ve min, max, mean, std = -3.1 0.76 0.34 0.28
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,8])), \
        '{0:.2g}'.format(np.max(metrics_list[:,8])), '{0:.2g}'.format(np.mean(metrics_list[:,8])), \
        '{0:.2g}'.format(np.std(metrics_list[:,8]))) #mpe_ve min, max, mean, std =  -0.067 0.84 0.57 0.11
#
print('mpe_ve min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_list[:,9])), \
        '{0:.2g}'.format(np.max(metrics_list[:,9])), '{0:.2g}'.format(np.mean(metrics_list[:,9])), \
        '{0:.2g}'.format(np.std(metrics_list[:,9]))) #mpe_ve min, max, mean, std =  1.9 32 22 6.1

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)