# Get DDS-enabled SWAT model calibration performance with respect to Obs data (No measurement errors) -- 5000 realz 
#   (500 realz/random-seed; we have 10 random seeds)
#
# Metrics --> R2-score, MSE, RMSE, NSE, logNSE, KGE, mKGE, npKGE, MPE--MeanPercentError (VolumeError)
#   1. Calibration Period --> Water Year (WY) from 2014 to 2016 -- Oct-1-2013 to Sep-30-2016 -- .iloc[6117:7213,:]
#   2. Validation Period  --> Water Year (WY) from 1999 to 2009 -- Oct-1-1999 to Sep-30-2009 -- .iloc[1003:4656,:]
#
#   CSV file: simulation periods: 1997/1/1 to 12/31/2016
#   https://www.timeanddate.com/date/durationresult.html?m1=1&d1=1&y1=1997&m2=12&d2=31&y2=2016&ti=on #7305 days
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
import datetime
from datetime import date, timedelta
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

#******************************************;
#  2a. Get dates (1997/1/1 to 12/31/2016)  ;
#******************************************;
sdate    = date(1997, 1, 1) # start date
edate    = date(2016, 12, 31) # end date
#
delta    = edate - sdate  # as timedelta
day_list = [] #days list
#
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    day_list.append(str(day))
    #print(i,day)

#**********************************************;
#  2b. Obs vs DDS model data (pre-processing)  ;
#**********************************************;
df_q_obs_cp  = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016 -- calibration period
df_q_obs_vp  = pd.read_csv(path_obs + 'obs_flow_wyval_data.csv', \
                        index_col = 0) #[3653 rows x 1 columns] WY-1999 to WY-2009 -- validation period
df_q_full    = pd.read_csv(path_swat + 'DDS/m0_flow_calb_param20_daymet_wy_2014_16_dds_kge500_random10_yr1997_2016.csv', \
                        index_col = 0) #[7305 rows x 5000 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_dds_cp   = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 5001 columns] WY-2014 to WY-2016
df_q_dds_vp   = df_q_copy.iloc[1003:4656,:].copy(deep=True) #[3653 rows x 5001 columns] WY-1999 to WY-2009
#
x_q_cp       = np.transpose(df_q_dds_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 1096)
x_q_vp       = np.transpose(df_q_dds_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 3653)
#
x_q_obs_cp   = df_q_obs_cp.values[:,0] #Obs. raw discharge (1096,)
x_q_obs_vp   = df_q_obs_vp.values[:,0] #Obs. raw discharge (3653,)

#*************************************************;
#  3a. SWAT q vs obs metrics (DDS models vs obs)  ;
#      Calibration period (WY-2014 to WY-2016)    ;
#*************************************************;
num_realz       = 5000 #DDS realz
metrics_cp_list = np.zeros((num_realz,10), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_cp_list)))
#
for i in range(0,num_realz):
    r2          = r2_score(x_q_obs_cp, x_q_cp[i,:]) #r2-score
    mse         = mean_squared_error(x_q_obs_cp, x_q_cp[i,:]) #MSE-score
    rmse        = mean_squared_error(x_q_obs_cp, x_q_cp[i,:], squared = False) #RMSE-score
    nse         = he.evaluator(he.nse, x_q_obs_cp, x_q_cp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_cp), np.log10(x_q_cp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_cp, x_q_cp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_cp, x_q_cp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_cp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    mpe_ve      = np.abs(100*(np.mean(x_q_cp[i,:]) - np.mean(x_q_obs_cp))/np.mean(x_q_obs_cp)) #Mean percent error or volume error
    print("dds-model, r2, mse, rmse, nse, lognse, kge, mkge, npkge, mpe_ve = ", i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve))
    #
    metrics_cp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
                        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve)

cols_list     = ['i', 'r2', 'mse', 'rmse', 'nse', 'lognse', 'kge', 'mkge', 'npkge', 'mpe_ve']
df_metrics_cp = pd.DataFrame(metrics_cp_list, columns = cols_list)
#
print('r2-score min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,1]))) #r2-score min, max, mean, std =  -5.3 0.62 0.36 0.39
#
print('mse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,2])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,2])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,2])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,2]))) #mse min, max, mean, std =  19 3.1e+02 31 19
#
print('rmse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,3])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,3])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,3])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,3]))) #rmse min, max, mean, std =  4.3 18 5.4 1.2
#
print('nse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,4])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,4])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,4])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,4]))) #nse min, max, mean, std =  -15 0.69 0.57 0.38
#
print('lognse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,5])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,5])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,5])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,5]))) #lognse min, max, mean, std =  -29 0.8 0.63 0.77
#
print('kge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,6])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,6])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,6])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,6]))) #kge min, max, mean, std =  -2.1 0.77 0.67 0.12
#
print('mkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,7])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,7])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,7])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,7]))) #mkge_ve min, max, mean, std = -2.7 0.79 0.71 0.14
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,8])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,8])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,8])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,8]))) #npkge_ve min, max, mean, std =  0.1 0.82 0.76 0.056
#
print('mpe_ve min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_cp_list[:,9])), \
        '{0:.2g}'.format(np.max(metrics_cp_list[:,9])), '{0:.2g}'.format(np.mean(metrics_cp_list[:,9])), \
        '{0:.2g}'.format(np.std(metrics_cp_list[:,9]))) #mpe_ve min, max, mean, std =  12 32 18 2.7

#*************************************************;
#  3b. SWAT q vs obs metrics (DDS models vs obs)  ;
#      Validation period (WY-1999 to WY-2009)     ;
#*************************************************;
num_realz       = 5000 #DDS realz
metrics_vp_list = np.zeros((num_realz,10), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_vp_list)))
#
for i in range(0,num_realz):
    r2          = r2_score(x_q_obs_vp, x_q_vp[i,:]) #r2-score
    mse         = mean_squared_error(x_q_obs_vp, x_q_vp[i,:]) #MSE-score
    rmse        = mean_squared_error(x_q_obs_vp, x_q_vp[i,:], squared = False) #RMSE-score
    nse         = he.evaluator(he.nse, x_q_obs_vp, x_q_vp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_vp), np.log10(x_q_vp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_vp, x_q_vp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_vp, x_q_vp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_vp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    mpe_ve      = np.abs(100*(np.mean(x_q_vp[i,:]) - np.mean(x_q_obs_vp))/np.mean(x_q_obs_vp)) #Mean percent error or volume error
    print("dds-model, r2, mse, rmse, nse, lognse, kge, mkge, npkge, mpe_ve = ", i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve))
    #
    metrics_vp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(mse), '{0:.2g}'.format(rmse), \
                        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge), '{0:.2g}'.format(mpe_ve)

cols_list     = ['i', 'r2', 'mse', 'rmse', 'nse', 'lognse', 'kge', 'mkge', 'npkge', 'mpe_ve']
df_metrics_vp = pd.DataFrame(metrics_vp_list, columns = cols_list)
#
print('r2-score min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,1]))) #r2-score min, max, mean, std =  -10 0.55 -1.3 1.3
#
print('mse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,2])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,2])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,2])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,2]))) #mse min, max, mean, std =  19 2.9e+02 38 17
#
print('rmse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,3])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,3])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,3])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,3]))) #rmse min, max, mean, std =  4.3 17 6 1.1
#
print('nse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,4])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,4])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,4])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,4]))) #nse min, max, mean, std =  -9 0.71 0.6 0.25
#
print('lognse min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,5])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,5])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,5])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,5]))) #lognse min, max, mean, std =  -13 0.77 0.5 0.4
#
print('kge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,6])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,6])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,6])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,6]))) #kge min, max, mean, std =  -1.2 0.77 0.6 0.096
#
print('mkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,7])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,7])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,7])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,7]))) #mkge_ve min, max, mean, std = -1.8 0.79 0.68 0.11
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,8])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,8])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,8])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,8]))) #npkge_ve min, max, mean, std =  0.054 0.81 0.68 0.058
#
print('mpe_ve min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_vp_list[:,9])), \
        '{0:.2g}'.format(np.max(metrics_vp_list[:,9])), '{0:.2g}'.format(np.mean(metrics_vp_list[:,9])), \
        '{0:.2g}'.format(np.std(metrics_vp_list[:,9]))) #mpe_ve min, max, mean, std =  3.5 39 18 4.5

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)