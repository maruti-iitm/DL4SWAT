# Descending order indices for DL, DDS, and GLUE performance metrics (Calibration and validation period)
#    1. DL set   (best 50 among 25050 realz)
#    2. DDS set  (best 50 among 5000 realz)
#    3. GLUE set (best 50 among 1000 realz)
#    4. Six different performance metrics --> R2-score, NSE, logNSE, KGE, mKGE, npKGE
#    5a. Calibration Period --> Water Year (WY) from 2014 to 2016 -- Oct-1-2013 to Sep-30-2016 -- .iloc[6117:7213,:]
#    5b. Validation Period  --> Water Year (WY) from 1999 to 2009 -- Oct-1-1999 to Sep-30-2009 -- .iloc[1003:4656,:]
#    6. Mean, std for GLUE behavioral parameter set based KGE >= 0.5
#
# https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
# arr1inds = arr1.argsort()
# sorted_arr1 = arr1[arr1inds[::-1]]
# sorted_arr2 = arr2[arr1inds[::-1]]
#
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import time
import spotpy #A Statistical Parameter Optimization Tool for Python
import pandas as pd
import numpy as np
import seaborn as sns
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

#******************************************;
#  2b. Get dates (1997/1/1 to 12/31/2020)  ;
#******************************************;
sdate1    = date(1997, 1, 1) # start date
edate1    = date(2020, 12, 31) # end date
#
delta1    = edate1 - sdate1  # as timedelta
day_list1 = [] #days list
#
for i in range(delta1.days + 1):
    day1 = sdate1 + timedelta(days=i)
    day_list1.append(str(day1))
    #print(i,day1)

#*********************************;
#  3a. Obs data (pre-processing)  ;
#*********************************;
df_q_obs_cp   = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016 -- calibration period
df_q_obs_vp   = pd.read_csv(path_obs + 'obs_flow_wyval_data.csv', \
                        index_col = 0) #[3653 rows x 1 columns] WY-1999 to WY-2009 -- validation period
#
x_q_obs_cp    = df_q_obs_cp.values[:,0] #Obs. raw discharge (1096,)
x_q_obs_vp    = df_q_obs_vp.values[:,0] #Obs. raw discharge (3653,)
#
cp_t_list     = df_q_obs_cp.index.to_list() #1096 
vp_t_list     = df_q_obs_vp.index.to_list() #3653

#*******************************************************;
#  3b. GLUE SWAT calib and valid data (pre-processing)  ;
#*******************************************************;
df_q_full    = pd.read_csv(path_swat + 'GLUE/m0_flow_glue_yr1997_2016_param20_5000.csv', \
                        index_col = 0) #[7305 rows x 5000 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_glue_cp = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 5001 columns] WY-2014 to WY-2016
df_q_glue_vp = df_q_copy.iloc[1003:4656,:].copy(deep=True) #[3653 rows x 5001 columns] WY-1999 to WY-2009
#
x_q_glue_cp  = np.transpose(df_q_glue_cp.iloc[:,1:].values + 0.00001)[0:1000,:] #Matrix of values of discharge -- 1000 realz; (1000, 1096)
x_q_glue_vp  = np.transpose(df_q_glue_vp.iloc[:,1:].values + 0.00001)[0:1000,:] #Matrix of values of discharge -- 1000 realz; (1000, 3653)
#
num_glue_realz       = 1000 #1000 GLUE realz
metrics_glue_cp_list = np.zeros((num_glue_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_glue_cp_list)))
#
for i in range(0,num_glue_realz):
    r2          = r2_score(x_q_obs_cp, x_q_glue_cp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_cp, x_q_glue_cp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_cp), np.log10(x_q_glue_cp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_cp, x_q_glue_cp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_cp, x_q_glue_cp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_glue_cp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("GLUE-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_glue_cp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_glue_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_glue_cp = pd.DataFrame(metrics_glue_cp_list, columns = cols_glue_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_glue_cp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_glue_cp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_glue_cp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_glue_cp_list[:,-1]))) #npkge min, max, mean, std =  -0.083 0.8 0.57 0.11
#np.argwhere(metrics_glue_cp_list[:,-1] == np.max(metrics_glue_cp_list[:,-1]))[:,0] #GLUE-realz = 640
#
num_glue_realz       = 1000 #1000 GLUE realz
metrics_glue_vp_list = np.zeros((num_glue_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_glue_vp_list)))
#
for i in range(0,num_glue_realz):
    r2          = r2_score(x_q_obs_vp, x_q_glue_vp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_vp, x_q_glue_vp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_vp), np.log10(x_q_glue_vp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_vp, x_q_glue_vp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_vp, x_q_glue_vp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_glue_vp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("GLUE-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_glue_vp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_glue_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_glue_vp = pd.DataFrame(metrics_glue_vp_list, columns = cols_glue_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_glue_vp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_glue_vp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_glue_vp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_glue_vp_list[:,-1]))) #npkge min, max, mean, std =  -0.047 0.8 0.5 0.11
#np.argwhere(metrics_glue_vp_list[:,-1] == np.max(metrics_glue_vp_list[:,-1]))[:,0] #GLUE-realz = 640
#
num_glue_realz       = 1000 #1000 GLUE realz
argsm_glue_cp_list   = np.zeros((num_glue_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
#arr1inds = metrics_glue_cp_list[:,1].argsort()[::-1]
#metrics_glue_cp_list[arr1inds,1]
#
argsm_glue_cp_list[:,0] = copy.deepcopy(metrics_glue_cp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_glue_cp_list[:,1] = copy.deepcopy(metrics_glue_cp_list[:,1].argsort()[::-1]) #R2-score indices
argsm_glue_cp_list[:,2] = copy.deepcopy(metrics_glue_cp_list[:,2].argsort()[::-1]) #NSE
argsm_glue_cp_list[:,3] = copy.deepcopy(metrics_glue_cp_list[:,3].argsort()[::-1]) #logNSE
argsm_glue_cp_list[:,4] = copy.deepcopy(metrics_glue_cp_list[:,4].argsort()[::-1]) #KGE
argsm_glue_cp_list[:,5] = copy.deepcopy(metrics_glue_cp_list[:,5].argsort()[::-1]) #mKGE
argsm_glue_cp_list[:,6] = copy.deepcopy(metrics_glue_cp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_glue_cp = pd.DataFrame(argsm_glue_cp_list, columns = cols_glue_list) #Sorted metrics-indices (descending)
#df_argsm_glue_cp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_cp.csv")
#
num_glue_realz       = 1000 #1000 GLUE realz
argsm_glue_vp_list   = np.zeros((num_glue_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
argsm_glue_vp_list[:,0] = copy.deepcopy(metrics_glue_vp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_glue_vp_list[:,1] = copy.deepcopy(metrics_glue_vp_list[:,1].argsort()[::-1]) #R2-score
argsm_glue_vp_list[:,2] = copy.deepcopy(metrics_glue_vp_list[:,2].argsort()[::-1]) #NSE
argsm_glue_vp_list[:,3] = copy.deepcopy(metrics_glue_vp_list[:,3].argsort()[::-1]) #logNSE
argsm_glue_vp_list[:,4] = copy.deepcopy(metrics_glue_vp_list[:,4].argsort()[::-1]) #KGE
argsm_glue_vp_list[:,5] = copy.deepcopy(metrics_glue_vp_list[:,5].argsort()[::-1]) #mKGE
argsm_glue_vp_list[:,6] = copy.deepcopy(metrics_glue_vp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_glue_vp = pd.DataFrame(argsm_glue_vp_list, columns = cols_glue_list) #Sorted metrics-indices (descending)
#df_argsm_glue_vp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_vp.csv")

#******************************************************;
#  3c. DDS SWAT calib and valid data (pre-processing)  ;
#******************************************************;
df_q_full    = pd.read_csv(path_swat + 'DDS/m0_flow_calb_param20_daymet_wy_2014_16_dds_kge500_random10_yr1997_2016.csv', \
                        index_col = 0) #[7305 rows x 5000 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_dds_cp  = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 5001 columns] WY-2014 to WY-2016
df_q_dds_vp  = df_q_copy.iloc[1003:4656,:].copy(deep=True) #[3653 rows x 5001 columns] WY-1999 to WY-2009
#
x_q_dds_cp   = np.transpose(df_q_dds_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 1096)
x_q_dds_vp   = np.transpose(df_q_dds_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 3653)
#
num_dds_realz       = 5000 #5000 DDS realz
metrics_dds_cp_list = np.zeros((num_dds_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dds_cp_list)))
#
for i in range(0,num_dds_realz):
    r2          = r2_score(x_q_obs_cp, x_q_dds_cp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_cp, x_q_dds_cp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_cp), np.log10(x_q_dds_cp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_cp, x_q_dds_cp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_cp, x_q_dds_cp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_dds_cp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("DDS-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_dds_cp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_dds_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_dds_cp = pd.DataFrame(metrics_dds_cp_list, columns = cols_dds_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dds_cp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_dds_cp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_dds_cp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_dds_cp_list[:,-1]))) #npkge min, max, mean, std =  0.1 0.82 0.76 0.056
#np.argwhere(metrics_dds_cp_list[:,-1] == np.max(metrics_dds_cp_list[:,-1]))[:,0] #DDS-realz = 229, 360, 392, 448, 486, 4621, 4794, 4880, 4967
#
num_dds_realz       = 5000 #5000 DDS realz
metrics_dds_vp_list = np.zeros((num_dds_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dds_vp_list)))
#
for i in range(0,num_dds_realz):
    r2          = r2_score(x_q_obs_vp, x_q_dds_vp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_vp, x_q_dds_vp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_vp), np.log10(x_q_dds_vp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_vp, x_q_dds_vp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_vp, x_q_dds_vp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_dds_vp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("DDS-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_dds_vp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_dds_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_dds_vp = pd.DataFrame(metrics_dds_vp_list, columns = cols_dds_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dds_vp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_dds_vp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_dds_vp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_dds_vp_list[:,-1]))) #npkge min, max, mean, std =  0.054 0.81 0.68 0.058
#np.argwhere(metrics_dds_vp_list[:,-1] == np.max(metrics_dds_vp_list[:,-1]))[:,0] #DDS-realz = 4621
#
num_dds_realz       = 5000 #5000 dds realz
argsm_dds_cp_list   = np.zeros((num_dds_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
#arr1inds = metrics_dds_cp_list[:,1].argsort()[::-1]
#metrics_dds_cp_list[arr1inds,1]
#
argsm_dds_cp_list[:,0] = copy.deepcopy(metrics_dds_cp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_dds_cp_list[:,1] = copy.deepcopy(metrics_dds_cp_list[:,1].argsort()[::-1]) #R2-score indices
argsm_dds_cp_list[:,2] = copy.deepcopy(metrics_dds_cp_list[:,2].argsort()[::-1]) #NSE
argsm_dds_cp_list[:,3] = copy.deepcopy(metrics_dds_cp_list[:,3].argsort()[::-1]) #logNSE
argsm_dds_cp_list[:,4] = copy.deepcopy(metrics_dds_cp_list[:,4].argsort()[::-1]) #KGE
argsm_dds_cp_list[:,5] = copy.deepcopy(metrics_dds_cp_list[:,5].argsort()[::-1]) #mKGE
argsm_dds_cp_list[:,6] = copy.deepcopy(metrics_dds_cp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_dds_cp = pd.DataFrame(argsm_dds_cp_list, columns = cols_dds_list) #Sorted metrics-indices (descending)
#df_argsm_dds_cp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_cp.csv")
#
num_dds_realz       = 5000 #5000 dds realz
argsm_dds_vp_list   = np.zeros((num_dds_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
argsm_dds_vp_list[:,0] = copy.deepcopy(metrics_dds_vp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_dds_vp_list[:,1] = copy.deepcopy(metrics_dds_vp_list[:,1].argsort()[::-1]) #R2-score
argsm_dds_vp_list[:,2] = copy.deepcopy(metrics_dds_vp_list[:,2].argsort()[::-1]) #NSE
argsm_dds_vp_list[:,3] = copy.deepcopy(metrics_dds_vp_list[:,3].argsort()[::-1]) #logNSE
argsm_dds_vp_list[:,4] = copy.deepcopy(metrics_dds_vp_list[:,4].argsort()[::-1]) #KGE
argsm_dds_vp_list[:,5] = copy.deepcopy(metrics_dds_vp_list[:,5].argsort()[::-1]) #mKGE
argsm_dds_vp_list[:,6] = copy.deepcopy(metrics_dds_vp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_dds_vp = pd.DataFrame(argsm_dds_vp_list, columns = cols_dds_list) #Sorted metrics-indices (descending)
#df_argsm_dds_vp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_vp.csv")

#************************************************************;
#  3d. DL-25050-realz calib and valid data (pre-processing)  ;
#************************************************************;
df_q_full    = pd.read_csv(path_swat + 'ML/m0_flow_ml_yr1997_2020_param20_top50.csv', \
                        index_col = 0) #[8766 rows x 50 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list1)
#
df_q_dl_cp   = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 51 columns] WY-2014 to WY-2016
df_q_dl_vp   = df_q_copy.iloc[1003:4656,:].copy(deep=True) #[3653 rows x 51 columns] WY-1999 to WY-2009
#
x_q_cp1      = np.transpose(df_q_dl_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 50 realz; (50, 1096)
x_q_vp1      = np.transpose(df_q_dl_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 50 realz; (50, 3653)
#
df_q_full    = pd.read_csv(path_swat + 'ML/m0_flow_ml_yr1997_2016_param20_25000.csv', \
                        index_col = 0) #[7305 rows x 25000 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_dl_cp   = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 25001 columns] WY-2014 to WY-2016
df_q_dl_vp   = df_q_copy.iloc[1003:4656,:].copy(deep=True) #[3653 rows x 25001 columns] WY-1999 to WY-2009
#
x_q_cp2      = np.transpose(df_q_dl_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 25000 realz; (25000, 1096)
x_q_vp2      = np.transpose(df_q_dl_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 25000 realz; (25000, 3653)
#
x_q_dl_cp    = np.concatenate((x_q_cp1, x_q_cp2)) #(25050, 1096)
x_q_dl_vp    = np.concatenate((x_q_vp1, x_q_vp2)) #(25050, 3653)
#
num_dl_realz       = 25050 #Top-50 models
metrics_dl_cp_list = np.zeros((num_dl_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dl_cp_list)))
#
for i in range(0,num_dl_realz):
    r2          = r2_score(x_q_obs_cp, x_q_dl_cp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_cp, x_q_dl_cp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_cp), np.log10(x_q_dl_cp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_cp, x_q_dl_cp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_cp, x_q_dl_cp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_dl_cp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("DL-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_dl_cp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_dl_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_dl_cp = pd.DataFrame(metrics_dl_cp_list, columns = cols_dl_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dl_cp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_dl_cp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_dl_cp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_dl_cp_list[:,-1]))) #npkge min, max, mean, std =  0.67 0.91 0.81 0.064
#np.argwhere(metrics_dl_cp_list[:,-1] == np.max(metrics_dl_cp_list[:,-1]))[:,0] #DL-realz = 3950 and 7950
#
num_dl_realz       = 25050 #Top-50 models
metrics_dl_vp_list = np.zeros((num_dl_realz,7), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dl_vp_list)))
#
for i in range(0,num_dl_realz):
    r2          = r2_score(x_q_obs_vp, x_q_dl_vp[i,:]) #r2-score
    nse         = he.evaluator(he.nse, x_q_obs_vp, x_q_dl_vp[i,:])[0] #NSE-score
    lognse      = he.evaluator(he.nse, np.log10(x_q_obs_vp), np.log10(x_q_dl_vp[i,:]))[0] #logNSE-score
    #
    kge, r, \
    alpha, beta = he.evaluator(he.kge, x_q_obs_vp, x_q_dl_vp[i,:]) #KGE-score
    #
    mkge, r, \
    gamma, beta = he.evaluator(he.kgeprime, x_q_obs_vp, x_q_dl_vp[i,:]) #mKGE-score
    #
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_dl_vp[i,:]) #npKGE-score
    #
    kge         = kge[0]
    mkge        = mkge[0]
    npkge       = npkge[0]
    print("DL-model, r2, nse, lognse, kge, mkge, npkge= ", i, '{0:.2g}'.format(r2), \
        '{0:.2g}'.format(nse), '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
        '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge))
    #
    metrics_dl_vp_list[i,:] = i, '{0:.2g}'.format(r2), '{0:.2g}'.format(nse), \
                                '{0:.2g}'.format(lognse), '{0:.2g}'.format(kge), \
                                '{0:.2g}'.format(mkge), '{0:.2g}'.format(npkge)

cols_dl_list     = ['i', 'r2', 'nse', 'lognse', 'kge', 'mkge', 'npkge']
df_metrics_dl_vp = pd.DataFrame(metrics_dl_vp_list, columns = cols_dl_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dl_vp_list[:,-1])), \
        '{0:.2g}'.format(np.max(metrics_dl_vp_list[:,-1])), '{0:.2g}'.format(np.mean(metrics_dl_vp_list[:,-1])), \
        '{0:.2g}'.format(np.std(metrics_dl_vp_list[:,-1]))) #npkge min, max, mean, std =  0.62 0.9 0.78 0.07
#np.argwhere(metrics_dl_vp_list[:,-1] == np.max(metrics_dl_vp_list[:,-1]))[:,0] #DL-realz = 2881, 6961, 8003, 10881
#
num_dl_realz       = 25050 #25050 dl realz
argsm_dl_cp_list   = np.zeros((num_dl_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
#arr1inds = metrics_dl_cp_list[:,1].argsort()[::-1]
#metrics_dl_cp_list[arr1inds,1]
#
argsm_dl_cp_list[:,0] = copy.deepcopy(metrics_dl_cp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_dl_cp_list[:,1] = copy.deepcopy(metrics_dl_cp_list[:,1].argsort()[::-1]) #R2-score indices
argsm_dl_cp_list[:,2] = copy.deepcopy(metrics_dl_cp_list[:,2].argsort()[::-1]) #NSE
argsm_dl_cp_list[:,3] = copy.deepcopy(metrics_dl_cp_list[:,3].argsort()[::-1]) #logNSE
argsm_dl_cp_list[:,4] = copy.deepcopy(metrics_dl_cp_list[:,4].argsort()[::-1]) #KGE
argsm_dl_cp_list[:,5] = copy.deepcopy(metrics_dl_cp_list[:,5].argsort()[::-1]) #mKGE
argsm_dl_cp_list[:,6] = copy.deepcopy(metrics_dl_cp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_dl_cp = pd.DataFrame(argsm_dl_cp_list, columns = cols_dl_list) #Sorted metrics-indices (descending)
#df_argsm_dl_cp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_cp.csv")
#
num_dl_realz       = 25050 #25050 dl realz
argsm_dl_vp_list   = np.zeros((num_dl_realz,7), dtype = float) #sorted-metrics-indices list (descending)
#
argsm_dl_vp_list[:,0] = copy.deepcopy(metrics_dl_vp_list[:,0]) #sorted numbers (not be confused with realz)
argsm_dl_vp_list[:,1] = copy.deepcopy(metrics_dl_vp_list[:,1].argsort()[::-1]) #R2-score
argsm_dl_vp_list[:,2] = copy.deepcopy(metrics_dl_vp_list[:,2].argsort()[::-1]) #NSE
argsm_dl_vp_list[:,3] = copy.deepcopy(metrics_dl_vp_list[:,3].argsort()[::-1]) #logNSE
argsm_dl_vp_list[:,4] = copy.deepcopy(metrics_dl_vp_list[:,4].argsort()[::-1]) #KGE
argsm_dl_vp_list[:,5] = copy.deepcopy(metrics_dl_vp_list[:,5].argsort()[::-1]) #mKGE
argsm_dl_vp_list[:,6] = copy.deepcopy(metrics_dl_vp_list[:,6].argsort()[::-1]) #npKGE
#
df_argsm_dl_vp = pd.DataFrame(argsm_dl_vp_list, columns = cols_dl_list) #Sorted metrics-indices (descending)
#df_argsm_dl_vp.to_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_vp.csv")

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)