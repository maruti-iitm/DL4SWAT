# Top-50 DL, DDS, and GLUE performance metrics boxplots (Calibration and validation period)
#    1. DL set   (best 50 among 25050 realz)
#    2. DDS set  (best 50 among 5000 realz)
#    3. GLUE set (best 50 among 1000 realz)
#    4. Six different performance metrics --> R2-score, NSE, logNSE, KGE, mKGE, npKGE
#    5a. Calibration Period --> Water Year (WY) from 2014 to 2016 -- Oct-1-2013 to Sep-30-2016 -- .iloc[6117:7213,:]
#    5b. Validation Period  --> Water Year (WY) from 1999 to 2013 -- Oct-1-1999 to Sep-30-2013 -- .iloc[1003:6117,:]
#    6. Mean, std for GLUE behavioral parameter set based KGE >= 0.5
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

#================================;
#  Function-1: Box plots colors  ; 
#================================;
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color)
    plt.setp(bp['medians'], color = color)
    plt.setp(bp['whiskers'], color = color)
    plt.setp(bp['caps'], color = color)

#===============================================================;
#  Function-2: Box plots for 6 different performance metrics    ; 
#===============================================================;
def plot_boxplot_metrics(x_pos, x_ticks_list, \
                             data1, data2, data3, data4, data5, data6, \
                             pos1, pos2, pos3, pos4, pos5, pos6, \
                             xmin, xmax, ymin, ymax, loc_pos, width_val, \
                             str_x_label, str_y_label, fig_name):

    #----------------------------------;
    #  Boxplot for metrics/error data  ;
    #----------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16, length = 6, width = 2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #
    bp1 = ax.boxplot(data1, positions = pos1, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp2 = ax.boxplot(data2, positions = pos2, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp3 = ax.boxplot(data3, positions = pos3, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp4 = ax.boxplot(data4, positions = pos4, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp5 = ax.boxplot(data5, positions = pos5, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp6 = ax.boxplot(data6, positions = pos6, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks_list)
    #
    set_box_color(bp1, color = 'black')
    set_box_color(bp2, color = 'black') 
    set_box_color(bp3, color = 'black') 
    set_box_color(bp4, color = 'black') 
    set_box_color(bp5, color = 'black')
    set_box_color(bp6, color = 'black')
    # 
    #colors = ['blue', 'green', 'red', ]
    colors = [[0,1,0,0.5],
              [0,0,1,0.5],
              [1,0,0,0.5]]
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        #patch.set_edgecolor('m')
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp5['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp6['boxes'], colors):
        patch.set_facecolor(color)
    hB,   = ax.plot([1],[1], color = colors[:][0], linewidth = 1.5)
    hG,   = ax.plot([1],[1], color = colors[:][1], linewidth = 1.5)
    hR,   = ax.plot([1],[1], color = colors[:][2], linewidth = 1.5)
    ax.legend((hB, hG, hR), ('CNN', 'DDS', 'GLUE'), loc = loc_pos)
    hR.set_visible(False)
    hG.set_visible(False)
    hB.set_visible(False)
    #tick_spacing = 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

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
df_q_obs_vp   = pd.read_csv(path_obs + 'obs_flow_wyval_data_1.csv', \
                        index_col = 0) #[5114 rows x 1 columns] WY-1999 to WY-2013 -- validation period
#
x_q_obs_cp    = df_q_obs_cp.values[:,0] #Obs. raw discharge (1096,)
x_q_obs_vp    = df_q_obs_vp.values[:,0] #Obs. raw discharge (5114,)
#
cp_t_list     = df_q_obs_cp.index.to_list() #1096 
vp_t_list     = df_q_obs_vp.index.to_list() #5114
#
q_obs_cp_fdc  = np.sort(x_q_obs_cp)[::-1] #(1096,)
q_obs_vp_fdc  = np.sort(x_q_obs_vp)[::-1] #(5114,)
#
exceedence_cp = np.arange(1.0,len(q_obs_cp_fdc)+1)/len(q_obs_cp_fdc) #(1096,)
exceedence_vp = np.arange(1.0,len(q_obs_vp_fdc)+1)/len(q_obs_vp_fdc) #(5114,)

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
df_q_glue_vp = df_q_copy.iloc[1003:6117,:].copy(deep=True) #[5114 rows x 5001 columns] WY-1999 to WY-2013
#
x_q_glue_cp  = np.transpose(df_q_glue_cp.iloc[:,1:].values + 0.00001)[0:1000,:] #Matrix of values of discharge -- 1000 realz; (1000, 1096)
x_q_glue_vp  = np.transpose(df_q_glue_vp.iloc[:,1:].values + 0.00001)[0:1000,:] #Matrix of values of discharge -- 1000 realz; (1000, 5114)
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
        '{0:.2g}'.format(np.std(metrics_glue_vp_list[:,-1]))) #npkge min, max, mean, std =  -0.068 0.79 0.48 0.11
#np.argwhere(metrics_glue_vp_list[:,-1] == np.max(metrics_glue_vp_list[:,-1]))[:,0] #GLUE-realz = 640
#
num_glue_realz       = 1000 #1000 GLUE realz
sm_glue_cp_list      = np.zeros((num_glue_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_glue_cp_list[:,0] = copy.deepcopy(metrics_glue_cp_list[:,0]) #sorted numbers (not be confused with realz)
sm_glue_cp_list[:,1] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,1])[::-1]) #R2-score
sm_glue_cp_list[:,2] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,2])[::-1]) #NSE
sm_glue_cp_list[:,3] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,3])[::-1]) #logNSE
sm_glue_cp_list[:,4] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,4])[::-1]) #KGE
sm_glue_cp_list[:,5] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,5])[::-1]) #mKGE
sm_glue_cp_list[:,6] = copy.deepcopy(np.sort(metrics_glue_cp_list[:,6])[::-1]) #npKGE
#
df_sm_glue_cp = pd.DataFrame(sm_glue_cp_list, columns = cols_glue_list) #Sorted metrics (descending)
#
num_glue_realz       = 1000 #1000 GLUE realz
sm_glue_vp_list      = np.zeros((num_glue_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_glue_vp_list[:,0] = copy.deepcopy(metrics_glue_vp_list[:,0]) #sorted numbers (not be confused with realz)
sm_glue_vp_list[:,1] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,1])[::-1]) #R2-score
sm_glue_vp_list[:,2] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,2])[::-1]) #NSE
sm_glue_vp_list[:,3] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,3])[::-1]) #logNSE
sm_glue_vp_list[:,4] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,4])[::-1]) #KGE
sm_glue_vp_list[:,5] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,5])[::-1]) #mKGE
sm_glue_vp_list[:,6] = copy.deepcopy(np.sort(metrics_glue_vp_list[:,6])[::-1]) #npKGE
#
df_sm_glue_vp = pd.DataFrame(sm_glue_vp_list, columns = cols_glue_list) #Sorted metrics (descending)
#
print('Behavior glue set (cp-mean), R2-score, NSE, logNSE, KGE, mKGE, npKGE = ', \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,1])), \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,2])), \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,3])), \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,4])), \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,5])), \
			'{0:.2g}'.format(np.mean(sm_glue_cp_list[0:172,6])))
print('Behavior glue set (cp-std), R2-score, NSE, logNSE, KGE, mKGE, npKGE = ', \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,1])), \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,2])), \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,3])), \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,4])), \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,5])), \
			'{0:.2g}'.format(np.std(sm_glue_cp_list[0:172,6])))
#np.argwhere(sm_glue_cp_list[:,1] >= 0.5)[:,0] #When R2-score > 0.5
#np.argwhere(sm_glue_cp_list[:,2] >= 0.5)[:,0] #When NSE > 0.5
#np.argwhere(sm_glue_cp_list[:,3] >= 0.5)[:,0] #When logNSE > 0.5
#np.argwhere(sm_glue_cp_list[:,4] >= 0.5)[:,0] #When KGE > 0.5
#np.argwhere(sm_glue_cp_list[:,5] >= 0.5)[:,0] #When mKGE > 0.5
#np.argwhere(sm_glue_cp_list[:,6] >= 0.5)[:,0] #When npKGE > 0.5
#
print('Behavior glue set (vp-mean), R2-score, NSE, logNSE, KGE, mKGE, npKGE = ', \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,1])), \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,2])), \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,3])), \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,4])), \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,5])), \
			'{0:.2g}'.format(np.mean(sm_glue_vp_list[0:141,6])))
print('Behavior glue set (vp-std), R2-score, NSE, logNSE, KGE, mKGE, npKGE = ', \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,1])), \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,2])), \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,3])), \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,4])), \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,5])), \
			'{0:.2g}'.format(np.std(sm_glue_vp_list[0:141,6])))
#np.argwhere(sm_glue_vp_list[:,1] >= 0.5)[:,0] #When R2-score > 0.5
#np.argwhere(sm_glue_vp_list[:,2] >= 0.5)[:,0] #When NSE > 0.5
#np.argwhere(sm_glue_vp_list[:,3] >= 0.5)[:,0] #When logNSE > 0.5
#np.argwhere(sm_glue_vp_list[:,4] >= 0.5)[:,0] #When KGE > 0.5
#np.argwhere(sm_glue_vp_list[:,5] >= 0.5)[:,0] #When mKGE > 0.5
#np.argwhere(sm_glue_vp_list[:,6] >= 0.5)[:,0] #When npKGE > 0.5

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
df_q_dds_vp  = df_q_copy.iloc[1003:6117,:].copy(deep=True) #[5114 rows x 5001 columns] WY-1999 to WY-2013
#
x_q_dds_cp   = np.transpose(df_q_dds_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 1096)
x_q_dds_vp   = np.transpose(df_q_dds_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 5000 realz; (5000, 5114)
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
        '{0:.2g}'.format(np.std(metrics_dds_vp_list[:,-1]))) #npkge min, max, mean, std =  0.026 0.8 0.68 0.058
#np.argwhere(metrics_dds_vp_list[:,-1] == np.max(metrics_dds_vp_list[:,-1]))[:,0] #DDS-realz = 4621
#
num_dds_realz       = 5000 #5000 DDS realz
sm_dds_cp_list      = np.zeros((num_dds_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_dds_cp_list[:,0] = copy.deepcopy(metrics_dds_cp_list[:,0]) #sorted numbers (not be confused with realz)
sm_dds_cp_list[:,1] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,1])[::-1]) #R2-score
sm_dds_cp_list[:,2] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,2])[::-1]) #NSE
sm_dds_cp_list[:,3] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,3])[::-1]) #logNSE
sm_dds_cp_list[:,4] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,4])[::-1]) #KGE
sm_dds_cp_list[:,5] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,5])[::-1]) #mKGE
sm_dds_cp_list[:,6] = copy.deepcopy(np.sort(metrics_dds_cp_list[:,6])[::-1]) #npKGE
#
df_sm_dds_cp = pd.DataFrame(sm_dds_cp_list, columns = cols_dds_list) #Sorted metrics (descending)
#
num_dds_realz       = 5000 #5000 DDS realz
sm_dds_vp_list      = np.zeros((num_dds_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_dds_vp_list[:,0] = copy.deepcopy(metrics_dds_vp_list[:,0]) #sorted numbers (not be confused with realz)
sm_dds_vp_list[:,1] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,1])[::-1]) #R2-score
sm_dds_vp_list[:,2] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,2])[::-1]) #NSE
sm_dds_vp_list[:,3] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,3])[::-1]) #logNSE
sm_dds_vp_list[:,4] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,4])[::-1]) #KGE
sm_dds_vp_list[:,5] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,5])[::-1]) #mKGE
sm_dds_vp_list[:,6] = copy.deepcopy(np.sort(metrics_dds_vp_list[:,6])[::-1]) #npKGE
#
df_sm_dds_vp = pd.DataFrame(sm_dds_vp_list, columns = cols_dds_list) #Sorted metrics (descending)

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
df_q_dl_vp   = df_q_copy.iloc[1003:6117,:].copy(deep=True) #[5114 rows x 51 columns] WY-1999 to WY-2013
#
x_q_cp1      = np.transpose(df_q_dl_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 50 realz; (50, 1096)
x_q_vp1      = np.transpose(df_q_dl_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 50 realz; (50, 5114)
#
df_q_full    = pd.read_csv(path_swat + 'ML/m0_flow_ml_yr1997_2016_param20_25000.csv', \
                        index_col = 0) #[7305 rows x 25000 columns]
print(np.argwhere(np.isnan(df_q_full.values)))
#
df_q_copy    = df_q_full.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_dl_cp   = df_q_copy.iloc[6117:7213,:].copy(deep=True) #[1096 rows x 25001 columns] WY-2014 to WY-2016
df_q_dl_vp   = df_q_copy.iloc[1003:6117,:].copy(deep=True) #[5114 rows x 25001 columns] WY-1999 to WY-2013
#
x_q_cp2      = np.transpose(df_q_dl_cp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 25000 realz; (25000, 1096)
x_q_vp2      = np.transpose(df_q_dl_vp.iloc[:,1:].values + 0.00001) #Matrix of values of discharge -- 25000 realz; (25000, 5114)
#
x_q_dl_cp    = np.concatenate((x_q_cp1, x_q_cp2)) #(25050, 1096)
x_q_dl_vp    = np.concatenate((x_q_vp1, x_q_vp2)) #(25050, 5114)
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
        '{0:.2g}'.format(np.std(metrics_dl_vp_list[:,-1]))) #npkge min, max, mean, std =  0.59 0.9 0.78 0.078
#np.argwhere(metrics_dl_vp_list[:,-1] == np.max(metrics_dl_vp_list[:,-1]))[:,0] #DL-realz = 2881, 6961, 8003, 10881
#
num_dl_realz       = 25050 #25050 DL realz
sm_dl_cp_list      = np.zeros((num_dl_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_dl_cp_list[:,0] = copy.deepcopy(metrics_dl_cp_list[:,0]) #sorted numbers (not be confused with realz)
sm_dl_cp_list[:,1] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,1])[::-1]) #R2-score
sm_dl_cp_list[:,2] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,2])[::-1]) #NSE
sm_dl_cp_list[:,3] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,3])[::-1]) #logNSE
sm_dl_cp_list[:,4] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,4])[::-1]) #KGE
sm_dl_cp_list[:,5] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,5])[::-1]) #mKGE
sm_dl_cp_list[:,6] = copy.deepcopy(np.sort(metrics_dl_cp_list[:,6])[::-1]) #npKGE
#
df_sm_dl_cp = pd.DataFrame(sm_dl_cp_list, columns = cols_dl_list) #Sorted metrics (descending)
#
num_dl_realz       = 25050 #25050 DL realz
sm_dl_vp_list      = np.zeros((num_dl_realz,7), dtype = float) #sorted-metrics list (descending)
#
sm_dl_vp_list[:,0] = copy.deepcopy(metrics_dl_vp_list[:,0]) #sorted numbers (not be confused with realz)
sm_dl_vp_list[:,1] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,1])[::-1]) #R2-score
sm_dl_vp_list[:,2] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,2])[::-1]) #NSE
sm_dl_vp_list[:,3] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,3])[::-1]) #logNSE
sm_dl_vp_list[:,4] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,4])[::-1]) #KGE
sm_dl_vp_list[:,5] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,5])[::-1]) #mKGE
sm_dl_vp_list[:,6] = copy.deepcopy(np.sort(metrics_dl_vp_list[:,6])[::-1]) #npKGE
#
df_sm_dl_vp = pd.DataFrame(sm_dl_vp_list, columns = cols_dl_list) #Sorted metrics (descending)

#*********************************************************;
#  4a. Performance metrics boxplots (Calibration period)  ;
#*********************************************************;
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'$R^2$', r'NSE', r'logNSE', r'KGE', r'mKGE', r'npKGE']
xmin         = 0
xmax         = 27
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'lower right'
str_x_label  = 'Performance metrics'
str_y_label  = 'Value'
fig_name     = path + 'Plots/SWAT_vs_Obs/Top50_Metrics_GLUE_DDS_CNN_cp'
#
pos1         = [2.0,  2.5,  3.0]
pos2         = [6.0,  6.5,  7.0]
pos3         = [10.0, 10.5, 11.0]
pos4         = [14.0, 14.5, 15.0]
pos5         = [18.0, 18.5, 19.0]     
pos6         = [22.0, 22.5, 23.0]   
#
test1        = [sm_dl_cp_list[0:100,1], sm_dds_cp_list[0:100,1], sm_glue_cp_list[0:100,1]] #R2-score
test2        = [sm_dl_cp_list[0:100,2], sm_dds_cp_list[0:100,2], sm_glue_cp_list[0:100,2]] #NSE
test3        = [sm_dl_cp_list[0:100,3], sm_dds_cp_list[0:100,3], sm_glue_cp_list[0:100,3]] #logNSE
test4        = [sm_dl_cp_list[0:100,4], sm_dds_cp_list[0:100,4], sm_glue_cp_list[0:100,4]] #KGE
test5        = [sm_dl_cp_list[0:100,5], sm_dds_cp_list[0:100,5], sm_glue_cp_list[0:100,5]] #mKGE
test6        = [sm_dl_cp_list[0:100,6], sm_dds_cp_list[0:100,6], sm_glue_cp_list[0:100,6]] #npKGE
#
plot_boxplot_metrics(x_pos, x_ticks_list, \
                        test1, test2, test3, test4, test5, test6, \
                        pos1, pos2, pos3, pos4, pos5, pos6, \
                        xmin, xmax, ymin, ymax, loc_pos, width_val, \
                        str_x_label, str_y_label, fig_name)

#********************************************************;
#  4b. Performance metrics boxplots (Validation period)  ;
#********************************************************;
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'$R^2$', r'NSE', r'logNSE', r'KGE', r'mKGE', r'npKGE']
xmin         = 0
xmax         = 27
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'lower right'
str_x_label  = 'Performance metrics'
str_y_label  = 'Value'
fig_name     = path + 'Plots/SWAT_vs_Obs/Top50_Metrics_GLUE_DDS_CNN_vp_1'
#
pos1         = [2.0,  2.5,  3.0]
pos2         = [6.0,  6.5,  7.0]
pos3         = [10.0, 10.5, 11.0]
pos4         = [14.0, 14.5, 15.0]
pos5         = [18.0, 18.5, 19.0]     
pos6         = [22.0, 22.5, 23.0]   
#
test1        = [sm_dl_vp_list[0:100,1], sm_dds_vp_list[0:100,1], sm_glue_vp_list[0:100,1]] #R2-score
test2        = [sm_dl_vp_list[0:100,2], sm_dds_vp_list[0:100,2], sm_glue_vp_list[0:100,2]] #NSE
test3        = [sm_dl_vp_list[0:100,3], sm_dds_vp_list[0:100,3], sm_glue_vp_list[0:100,3]] #logNSE
test4        = [sm_dl_vp_list[0:100,4], sm_dds_vp_list[0:100,4], sm_glue_vp_list[0:100,4]] #KGE
test5        = [sm_dl_vp_list[0:100,5], sm_dds_vp_list[0:100,5], sm_glue_vp_list[0:100,5]] #mKGE
test6        = [sm_dl_vp_list[0:100,6], sm_dds_vp_list[0:100,6], sm_glue_vp_list[0:100,6]] #npKGE
#
plot_boxplot_metrics(x_pos, x_ticks_list, \
                        test1, test2, test3, test4, test5, test6, \
                        pos1, pos2, pos3, pos4, pos5, pos6, \
                        xmin, xmax, ymin, ymax, loc_pos, width_val, \
                        str_x_label, str_y_label, fig_name)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)