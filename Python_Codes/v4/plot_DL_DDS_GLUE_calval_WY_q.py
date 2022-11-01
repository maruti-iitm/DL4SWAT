# Calibration and Validation period plots include
#   1a. Calibration Period --> Water Year (WY) from 2014 to 2016 -- Oct-1-2013 to Sep-30-2016 -- .iloc[6117:7213,:]
#   1b. Validation Period  --> Water Year (WY) from 1999 to 2009 -- Oct-1-1999 to Sep-30-2009 -- .iloc[1003:4656,:]
#	2. One-to-one scatterplots, mean variation of streamflow vs. time period, 
#	3. Flow duration curves, streamflow time-series
#	4. Methods used:
#		a. DL   --> 50 sets + 25000 sets
#		b. GLUE --> 1000 sets
#		c. DDS  --> 10*500 = 5000 sets
#	4. Probability that observational data is contained within the prediction bounds estimated by DL/GLUE/DDS
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
np.set_printoptions(precision=2)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#==============================================;
#  Function-1: Plot one-to-one for best q-run  ; 
#==============================================;
def plot_gt_pred(x, y, param_id, fl_name, \
                 str_x_label, str_y_label, col):

    #------------------------------------------------;
    #  Plot one-to-one (ground truth vs. predicted)  ;
    #------------------------------------------------;
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
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    min_val = np.min(x)
    max_val = np.max(x)
    #
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    sns.lineplot([min_val, max_val], [min_val, max_val], \
                  linestyle = 'solid', linewidth = 1.5, \
                  color = col) #One-to-One line
    sns.scatterplot(x, y, color = col, marker = 'o')
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(fl_name + str(param_id) + '.pdf')
    fig.savefig(fl_name + str(param_id) + '.png')
    plt.close(fig)

#==============================================================================;
#  Function-2: Plot obs vs. best q-run + confidence interval (CNN, GLUE, DDS)  ;
#==============================================================================;
def plot_obs_vs_swat_data(t_list, y_mean, y_lb, y_ub, y_obs, \
                          str_x_label, str_y_label, fl_name, \
                          legend_loc, obs_label_list, num_ticks, \
                          line_width_ens, alpha_val, col_list):

    #------------------------------------------;
    #  Obs data vs. mean ens data +/- 2 * std  ;
    #------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure(figsize=(20,5))
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 18)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([0, 730])
    ax.set_ylim([0, 80])
    sns.lineplot(t_list, y_obs, linestyle = 'solid', linewidth = line_width_ens[0], color = col_list[0], \
                 marker = None, label = obs_label_list[0]) #Obs. data
    sns.lineplot(t_list, y_mean, linestyle = 'dashed', linewidth = line_width_ens[1], color = col_list[1], \
                 marker = None, label = obs_label_list[1]) #Best data
    ax.fill_between(t_list, y_lb, y_ub, linestyle = 'solid', linewidth = 0.5, \
                    color = col_list[2], alpha = alpha_val) #Mean +/ 2*std or 95% CI
    tick_spacing = num_ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = legend_loc)
    fig.tight_layout()
    fig.savefig(fl_name + '.pdf')
    fig.savefig(fl_name + '.png', dpi = 300) #Medium resolution
    plt.close(fig)

#=================================================================;
#  Function-3: Plot flow duration curve (Obs vs. SWAT ensembles)  ;
#=================================================================;
def plot_fdc_lines(x_list, y_ens, y_mean, y_obs, str_x_label, str_y_label, \
                    fig_name, num_realz, \
                    legend_loc, plot_label_list, num_ticks,  \
                    line_width_ens, alpha_val, col_list):

    #------------------------------------------;
    #  Obs data vs. mean ens data +/- 2 * std  ;
    #------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 18)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6, width = 2)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6, width = 2)
    ax.set_xlim([0, 100])
    #ax.set_ylim([0, 80])
    for i in range(0,num_realz):
        if i == 0:
            ax.plot(x_list, y_ens[:,i], linestyle = 'solid', \
                        linewidth = line_width_ens[2], color = col_list[2], \
                        marker = None, label = plot_label_list[2], \
                        alpha = alpha_val) #SWAT ensemble discharge FDC
        else:
            ax.plot(x_list, y_ens[:,i], linestyle = 'solid', \
                        linewidth = line_width_ens[2], color = col_list[2], \
                        marker = None, alpha = alpha_val) #SWAT ensemble discharge FDC
    ax.plot(x_list, y_obs, linestyle = 'solid', linewidth = line_width_ens[0], color = col_list[0], \
                 marker = None, label = plot_label_list[0]) #Obs. discharge FDC
    ax.plot(x_list, y_mean, linestyle = 'dashed', linewidth = line_width_ens[1], color = col_list[1], \
                 marker = None, label = plot_label_list[1]) #Mean discharge FDC
    ax.set_yscale('log')
    tick_spacing = num_ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = legend_loc)
    #ax.set_aspect(1./ax.get_data_ratio())
    fig.tight_layout()
    #fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

#=======================================================================;
#  Function-4: Metrics for prob and size of mean variation (Histogram)  ;
#=======================================================================;
def plot_mv_prob_metrics(x_list, y_list, xticks_list, xticks_label_list, \
    ymin, ymax, loc_pos, str_x_label, str_y_label, fig_name):

    #------------------------------------------------;
    #  Metrics and prob for calib and valid periods  ;
    #------------------------------------------------;
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
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #
    ax.bar(x_list[0], y_list[0], align = 'center', \
        width = width_val, color = 'blue', edgecolor = 'k', alpha = 0.5, label = 'CNN')
    ax.bar(x_list[1], y_list[1], align = 'center', \
        width = width_val, color = 'green', edgecolor = 'k', alpha = 0.5, label = 'DDS')
    ax.bar(x_list[2], y_list[2], align = 'center', \
        width = width_val, color = 'red', edgecolor = 'k', alpha = 0.5, label = 'GLUE')
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label_list)
    #tick_spacing = 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
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

#************************************************;
#  2c. Read the descending order sorted indices  ;
#************************************************;
df_argsm_glue_cp = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_cp.csv", \
                                index_col = 0, dtype = int) #[1000 rows x 7 columns]
df_argsm_glue_vp = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_vp.csv", \
                                index_col = 0, dtype = int) #[1000 rows x 7 columns]
df_argsm_dds_cp  = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_cp.csv", \
                                index_col = 0, dtype = int) #[5000 rows x 7 columns]
df_argsm_dds_vp  = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_vp.csv", \
                                index_col = 0, dtype = int) #[5000 rows x 7 columns]
df_argsm_dl_cp   = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_cp.csv", \
                                index_col = 0, dtype = int) #[25050 rows x 7 columns]
df_argsm_dl_vp   = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_vp.csv", \
                                index_col = 0, dtype = int) #[25050 rows x 7 columns]

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
#
q_obs_cp_fdc  = np.sort(x_q_obs_cp)[::-1] #(1096,)
q_obs_vp_fdc  = np.sort(x_q_obs_vp)[::-1] #(3653,)
#
exceedence_cp = np.arange(1.0,len(q_obs_cp_fdc)+1)/len(q_obs_cp_fdc) #(1096,)
exceedence_vp = np.arange(1.0,len(q_obs_vp_fdc)+1)/len(q_obs_vp_fdc) #(3653,)

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
metrics_glue_cp_list = np.zeros((num_glue_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_glue_cp_list)))
#
for i in range(0,num_glue_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_glue_cp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("GLUE-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_glue_cp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_glue_list     = ['i',  'npkge']
df_metrics_glue_cp = pd.DataFrame(metrics_glue_cp_list, columns = cols_glue_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_glue_cp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_glue_cp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_glue_cp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_glue_cp_list[:,1]))) #npkge min, max, mean, std =  -0.083 0.8 0.57 0.11
#np.argwhere(metrics_glue_cp_list[:,1] == np.max(metrics_glue_cp_list[:,1]))[:,0] #GLUE-realz = 640
#
num_glue_realz       = 1000 #1000 GLUE realz
metrics_glue_vp_list = np.zeros((num_glue_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_glue_vp_list)))
#
for i in range(0,num_glue_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_glue_vp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("GLUE-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_glue_vp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_glue_list     = ['i',  'npkge']
df_metrics_glue_vp = pd.DataFrame(metrics_glue_vp_list, columns = cols_glue_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_glue_vp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_glue_vp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_glue_vp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_glue_vp_list[:,1]))) #npkge min, max, mean, std =  -0.047 0.8 0.5 0.11
#np.argwhere(metrics_glue_vp_list[:,1] == np.max(metrics_glue_vp_list[:,1]))[:,0] #GLUE-realz = 640
#
q_glue_cp_fdc      = np.sort(x_q_glue_cp.T, axis = 0)[::-1] #(1096, 1000)
q_glue_best_cp_fdc = np.sort(x_q_glue_cp[640,:])[::-1] #(1096,)
#
q_glue_vp_fdc      = np.sort(x_q_glue_vp.T, axis = 0)[::-1] #(3653, 1000)
q_glue_best_vp_fdc = np.sort(x_q_glue_vp[640,:])[::-1] #(3653,)

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
metrics_dds_cp_list = np.zeros((num_dds_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dds_cp_list)))
#
for i in range(0,num_dds_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_dds_cp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("DDS-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_dds_cp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_dds_list     = ['i',  'npkge']
df_metrics_dds_cp = pd.DataFrame(metrics_dds_cp_list, columns = cols_dds_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dds_cp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_dds_cp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_dds_cp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_dds_cp_list[:,1]))) #npkge min, max, mean, std =  0.1 0.82 0.76 0.056
#np.argwhere(metrics_dds_cp_list[:,1] == np.max(metrics_dds_cp_list[:,1]))[:,0] #DDS-realz = 229, 360, 392, 448, 486, 4621, 4794, 4880, 4967
#
num_dds_realz       = 5000 #5000 DDS realz
metrics_dds_vp_list = np.zeros((num_dds_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dds_vp_list)))
#
for i in range(0,num_dds_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_dds_vp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("DDS-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_dds_vp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_dds_list     = ['i',  'npkge']
df_metrics_dds_vp = pd.DataFrame(metrics_dds_vp_list, columns = cols_dds_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dds_vp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_dds_vp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_dds_vp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_dds_vp_list[:,1]))) #npkge min, max, mean, std =  0.054 0.81 0.68 0.058
#np.argwhere(metrics_dds_vp_list[:,1] == np.max(metrics_dds_vp_list[:,1]))[:,0] #DDS-realz = 4621
#
q_dds_cp_fdc      = np.sort(x_q_dds_cp.T, axis = 0)[::-1] #(1096, 1000)
q_dds_best_cp_fdc = np.sort(x_q_dds_cp[229,:])[::-1] #(1096,)
#
q_dds_vp_fdc      = np.sort(x_q_dds_vp.T, axis = 0)[::-1] #(3653, 1000)
q_dds_best_vp_fdc = np.sort(x_q_dds_vp[4621,:])[::-1] #(3653,)

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
metrics_dl_cp_list = np.zeros((num_dl_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dl_cp_list)))
#
for i in range(0,num_dl_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_cp, x_q_dl_cp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("DL-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_dl_cp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_dl_list     = ['i',  'npkge']
df_metrics_dl_cp = pd.DataFrame(metrics_dl_cp_list, columns = cols_dl_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dl_cp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_dl_cp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_dl_cp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_dl_cp_list[:,1]))) #npkge min, max, mean, std =  0.67 0.91 0.81 0.064
#np.argwhere(metrics_dl_cp_list[:,1] == np.max(metrics_dl_cp_list[:,1]))[:,0] #DL-realzs = 3950 and 7950
#
num_dl_realz       = 25050 #Top-50 models
metrics_dl_vp_list = np.zeros((num_dl_realz,2), dtype = float) #metrics list
print(np.argwhere(np.isnan(metrics_dl_vp_list)))
#
for i in range(0,num_dl_realz):
    npkge, r, \
    alpha, beta = he.evaluator(he.kgenp, x_q_obs_vp, x_q_dl_vp[i,:]) #npKGE-score
    #
    npkge       = npkge[0]
    print("DL-model, npkge = ", i, '{0:.2g}'.format(npkge))
    #
    metrics_dl_vp_list[i,:] = i, '{0:.2g}'.format(npkge)

cols_dl_list     = ['i',  'npkge']
df_metrics_dl_vp = pd.DataFrame(metrics_dl_vp_list, columns = cols_dl_list)
#
print('npkge min, max, mean, std = ', '{0:.2g}'.format(np.min(metrics_dl_vp_list[:,1])), \
        '{0:.2g}'.format(np.max(metrics_dl_vp_list[:,1])), '{0:.2g}'.format(np.mean(metrics_dl_vp_list[:,1])), \
        '{0:.2g}'.format(np.std(metrics_dl_vp_list[:,1]))) #npkge min, max, mean, std =  0.62 0.9 0.78 0.07
#np.argwhere(metrics_dl_vp_list[:,1] == np.max(metrics_dl_vp_list[:,1]))[:,0] #DL-realzs = 2881, 6961, 8003, 10881
#
q_dl_cp_fdc      = np.sort(x_q_dl_cp.T, axis = 0)[::-1] #(1096, 1000)
q_dl_best_cp_fdc = np.sort(x_q_dl_cp[3950,:])[::-1] #(1096,)
#
q_dl_vp_fdc      = np.sort(x_q_dl_vp.T, axis = 0)[::-1] #(3653, 1000)
q_dl_best_vp_fdc = np.sort(x_q_dl_vp[2881,:])[::-1] #(3653,)

#*************************************************************;
#  4a. Percent errors and obs within the uncertainity (glue)  ;
#*************************************************************;
q_glue_cp_err_list   = np.zeros((len(cp_t_list),num_glue_realz), dtype = float) #Percent error list for glue; #(1096, 25050)
q_glue_cp_uncer_list = np.zeros(len(cp_t_list), dtype = int) #Obs data within the uncertainity boundary (glue) #(1096,)
q_glue_cp_lbub_list  = np.zeros((len(cp_t_list),3), dtype = float) #lbub list for glue #(1096, 3)
#
for i in range(0,num_glue_realz):
    q_glue_cp_percent_error = np.abs(np.divide(x_q_obs_cp - x_q_glue_cp[i,:], x_q_obs_cp))*100 #percent error
    q_glue_cp_err_list[:,i] = copy.deepcopy(q_glue_cp_percent_error)

for i in range(0,len(cp_t_list)):
    q_glue_cp_lb              = np.min(x_q_glue_cp[:,i]) #Min among each day
    q_glue_cp_ub              = np.max(x_q_glue_cp[:,i]) #Max among each day
    q_glue_cp_lbub_list[i,0]  = q_glue_cp_lb
    q_glue_cp_lbub_list[i,1]  = q_glue_cp_ub
    q_glue_cp_lbub_list[i,2]  = q_glue_cp_ub - q_glue_cp_lb
    #
    if x_q_obs_cp[i] <= q_glue_cp_ub and x_q_obs_cp[i] >= q_glue_cp_lb:
        print(i, q_glue_cp_lb, q_glue_cp_ub, x_q_obs_cp[i])
        q_glue_cp_uncer_list[i] = 1

q_glue_cp_mean_variation = (1.0/len(cp_t_list)) * np.sum(q_glue_cp_lbub_list[:,2]) #31.43770912408759
q_glue_cp_prob           = (1.0/len(cp_t_list)) * np.sum(q_glue_cp_uncer_list) #0.9826642335766423
#
q_glue_vp_err_list   = np.zeros((len(vp_t_list),num_glue_realz), dtype = float) #Percent error list for glue; #(3653, 25050)
q_glue_vp_uncer_list = np.zeros(len(vp_t_list), dtype = int) #Obs data within the uncertainity boundary (glue) #(3653,)
q_glue_vp_lbub_list  = np.zeros((len(vp_t_list),3), dtype = float) #lbub list for glue #(3653, 3)
#
for i in range(0,num_glue_realz):
    q_glue_vp_percent_error = np.abs(np.divide(x_q_obs_vp - x_q_glue_vp[i,:], x_q_obs_vp))*100 #percent error
    q_glue_vp_err_list[:,i] = copy.deepcopy(q_glue_vp_percent_error)

for i in range(0,len(vp_t_list)):
    q_glue_vp_lb              = np.min(x_q_glue_vp[:,i]) #Min among each day
    q_glue_vp_ub              = np.max(x_q_glue_vp[:,i]) #Max among each day
    q_glue_vp_lbub_list[i,0]  = q_glue_vp_lb
    q_glue_vp_lbub_list[i,1]  = q_glue_vp_ub
    q_glue_vp_lbub_list[i,2]  = q_glue_vp_ub - q_glue_vp_lb
    #
    if x_q_obs_vp[i] <= q_glue_vp_ub and x_q_obs_vp[i] >= q_glue_vp_lb:
        print(i, q_glue_vp_lb, q_glue_vp_ub, x_q_obs_vp[i])
        q_glue_vp_uncer_list[i] = 1

q_glue_vp_mean_variation = (1.0/len(vp_t_list)) * np.sum(q_glue_vp_lbub_list[:,2]) #31.199448353134407
q_glue_vp_prob           = (1.0/len(vp_t_list)) * np.sum(q_glue_vp_uncer_list) #0.993430057486997

#************************************************************;
#  4b. Percent errors and obs within the uncertainity (dds)  ;
#************************************************************;
q_dds_cp_err_list   = np.zeros((len(cp_t_list),num_dds_realz), dtype = float) #Percent error list for dds; #(1096, 25050)
q_dds_cp_uncer_list = np.zeros(len(cp_t_list), dtype = int) #Obs data within the uncertainity boundary (dds) #(1096,)
q_dds_cp_lbub_list  = np.zeros((len(cp_t_list),3), dtype = float) #lbub list for dds #(1096, 3)
#
for i in range(0,num_dds_realz):
    q_dds_cp_percent_error = np.abs(np.divide(x_q_obs_cp - x_q_dds_cp[i,:], x_q_obs_cp))*100 #percent error
    q_dds_cp_err_list[:,i] = copy.deepcopy(q_dds_cp_percent_error)

for i in range(0,len(cp_t_list)):
    q_dds_cp_lb              = np.min(x_q_dds_cp[:,i]) #Min among each day
    q_dds_cp_ub              = np.max(x_q_dds_cp[:,i]) #Max among each day
    q_dds_cp_lbub_list[i,0]  = q_dds_cp_lb
    q_dds_cp_lbub_list[i,1]  = q_dds_cp_ub
    q_dds_cp_lbub_list[i,2]  = q_dds_cp_ub - q_dds_cp_lb
    #
    if x_q_obs_cp[i] <= q_dds_cp_ub and x_q_obs_cp[i] >= q_dds_cp_lb:
        print(i, q_dds_cp_lb, q_dds_cp_ub, x_q_obs_cp[i])
        q_dds_cp_uncer_list[i] = 1

q_dds_cp_mean_variation = (1.0/len(cp_t_list)) * np.sum(q_dds_cp_lbub_list[:,2]) #26.301152756386863
q_dds_cp_prob           = (1.0/len(cp_t_list)) * np.sum(q_dds_cp_uncer_list) #0.9872262773722628
#
q_dds_vp_err_list   = np.zeros((len(vp_t_list),num_dds_realz), dtype = float) #Percent error list for dds; #(3653, 25050)
q_dds_vp_uncer_list = np.zeros(len(vp_t_list), dtype = int) #Obs data within the uncertainity boundary (dds) #(3653,)
q_dds_vp_lbub_list  = np.zeros((len(vp_t_list),3), dtype = float) #lbub list for dds #(3653, 3)
#
for i in range(0,num_dds_realz):
    q_dds_vp_percent_error = np.abs(np.divide(x_q_obs_vp - x_q_dds_vp[i,:], x_q_obs_vp))*100 #percent error
    q_dds_vp_err_list[:,i] = copy.deepcopy(q_dds_vp_percent_error)

for i in range(0,len(vp_t_list)):
    q_dds_vp_lb              = np.min(x_q_dds_vp[:,i]) #Min among each day
    q_dds_vp_ub              = np.max(x_q_dds_vp[:,i]) #Max among each day
    q_dds_vp_lbub_list[i,0]  = q_dds_vp_lb
    q_dds_vp_lbub_list[i,1]  = q_dds_vp_ub
    q_dds_vp_lbub_list[i,2]  = q_dds_vp_ub - q_dds_vp_lb
    #
    if x_q_obs_vp[i] <= q_dds_vp_ub and x_q_obs_vp[i] >= q_dds_vp_lb:
        print(i, q_dds_vp_lb, q_dds_vp_ub, x_q_obs_vp[i])
        q_dds_vp_uncer_list[i] = 1

q_dds_vp_mean_variation = (1.0/len(vp_t_list)) * np.sum(q_dds_vp_lbub_list[:,2]) #25.496431042417193
q_dds_vp_prob           = (1.0/len(vp_t_list)) * np.sum(q_dds_vp_uncer_list) #0.9920613194634547

#***********************************************************;
#  4c. Percent errors and obs within the uncertainity (dl)  ;
#***********************************************************;
q_dl_cp_err_list   = np.zeros((len(cp_t_list),num_dl_realz), dtype = float) #Percent error list for dl; #(1096, 25050)
q_dl_cp_uncer_list = np.zeros(len(cp_t_list), dtype = int) #Obs data within the uncertainity boundary (dl) #(1096,)
q_dl_cp_lbub_list  = np.zeros((len(cp_t_list),3), dtype = float) #lbub list for dl #(1096, 3)
#
for i in range(0,num_dl_realz):
    q_dl_cp_percent_error = np.abs(np.divide(x_q_obs_cp - x_q_dl_cp[i,:], x_q_obs_cp))*100 #percent error
    q_dl_cp_err_list[:,i] = copy.deepcopy(q_dl_cp_percent_error)

for i in range(0,len(cp_t_list)):
    q_dl_cp_lb              = np.min(x_q_dl_cp[:,i]) #Min among each day
    q_dl_cp_ub              = np.max(x_q_dl_cp[:,i]) #Max among each day
    q_dl_cp_lbub_list[i,0]  = q_dl_cp_lb
    q_dl_cp_lbub_list[i,1]  = q_dl_cp_ub
    q_dl_cp_lbub_list[i,2]  = q_dl_cp_ub - q_dl_cp_lb
    #
    if x_q_obs_cp[i] <= q_dl_cp_ub and x_q_obs_cp[i] >= q_dl_cp_lb:
        print(i, q_dl_cp_lb, q_dl_cp_ub, x_q_obs_cp[i])
        q_dl_cp_uncer_list[i] = 1

q_dl_cp_mean_variation = (1.0/len(cp_t_list)) * np.sum(q_dl_cp_lbub_list[:,2]) #14.590690959279197
q_dl_cp_prob           = (1.0/len(cp_t_list)) * np.sum(q_dl_cp_uncer_list) #0.9680656934306568
#
q_dl_vp_err_list   = np.zeros((len(vp_t_list),num_dl_realz), dtype = float) #Percent error list for dl; #(3653, 25050)
q_dl_vp_uncer_list = np.zeros(len(vp_t_list), dtype = int) #Obs data within the uncertainity boundary (dl) #(3653,)
q_dl_vp_lbub_list  = np.zeros((len(vp_t_list),3), dtype = float) #lbub list for dl #(3653, 3)
#
for i in range(0,num_dl_realz):
    q_dl_vp_percent_error = np.abs(np.divide(x_q_obs_vp - x_q_dl_vp[i,:], x_q_obs_vp))*100 #percent error
    q_dl_vp_err_list[:,i] = copy.deepcopy(q_dl_vp_percent_error)

for i in range(0,len(vp_t_list)):
    q_dl_vp_lb              = np.min(x_q_dl_vp[:,i]) #Min among each day
    q_dl_vp_ub              = np.max(x_q_dl_vp[:,i]) #Max among each day
    q_dl_vp_lbub_list[i,0]  = q_dl_vp_lb
    q_dl_vp_lbub_list[i,1]  = q_dl_vp_ub
    q_dl_vp_lbub_list[i,2]  = q_dl_vp_ub - q_dl_vp_lb
    #
    if x_q_obs_vp[i] <= q_dl_vp_ub and x_q_obs_vp[i] >= q_dl_vp_lb:
        print(i, q_dl_vp_lb, q_dl_vp_ub, x_q_obs_vp[i])
        q_dl_vp_uncer_list[i] = 1

q_dl_vp_mean_variation = (1.0/len(vp_t_list)) * np.sum(q_dl_vp_lbub_list[:,2]) #14.03329540719956
q_dl_vp_prob           = (1.0/len(vp_t_list)) * np.sum(q_dl_vp_uncer_list) #0.9937038050917054

#*********************************************************************;
#  5a. Plot best DL, GLUE, and DDS scatter plot (calibration period)  ;
#*********************************************************************;
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (GLUE)'
param_id    = 'GLUE_cp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'red'
plot_gt_pred(x_q_obs_cp, x_q_glue_cp[640,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)
#
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (DDS)'
param_id    = 'DDS_cp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'green'
plot_gt_pred(x_q_obs_cp, x_q_dds_cp[229,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)
#
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (CNN)'
param_id    = 'CNN_cp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'blue'
plot_gt_pred(x_q_obs_cp, x_q_dl_cp[3950,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)

#********************************************************************;
#  5b. Plot best DL, GLUE, and DDS scatter plot (validation period)  ;
#********************************************************************;
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (GLUE)'
param_id    = 'GLUE_vp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'red'
plot_gt_pred(x_q_obs_vp, x_q_glue_vp[640,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)
#
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (DDS)'
param_id    = 'DDS_vp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'green'
plot_gt_pred(x_q_obs_vp, x_q_dds_vp[4621,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)
#
str_x_label = r'Observational data [$\frac{m^3}{s}$]'
str_y_label = r'Calibrated SWAT (CNN)'
param_id    = 'CNN_vp'
fl_name     = path + 'Plots/SWAT_vs_Obs/SWAT_vs_Obs_'
col         = 'blue'
plot_gt_pred(x_q_obs_vp, x_q_dl_vp[2881,:], param_id, fl_name, \
                 str_x_label, str_y_label, col)

#****************************************************************************************;
#  6a. Confidence intervals for DL, GLUE, and DDS time-series (calib and valid periods)  ;
#****************************************************************************************;
q_glue_cp_lb_list = x_q_glue_cp.min(axis = 0) #(1096,)
q_glue_cp_ub_list = x_q_glue_cp.max(axis = 0) #(1096,)
#
q_glue_vp_lb_list = x_q_glue_vp.min(axis = 0) #(3653,)
q_glue_vp_ub_list = x_q_glue_vp.max(axis = 0) #(3653,)
#
q_dds_cp_lb_list  = x_q_dds_cp.min(axis = 0) #(1096,)
q_dds_cp_ub_list  = x_q_dds_cp.max(axis = 0) #(1096,)
#
q_dds_vp_lb_list  = x_q_dds_vp.min(axis = 0) #(3653,)
q_dds_vp_ub_list  = x_q_dds_vp.max(axis = 0) #(3653,)
#
q_dl_cp_lb_list   = x_q_dl_cp.min(axis = 0) #(1096,)
q_dl_cp_ub_list   = x_q_dl_cp.max(axis = 0) #(1096,)
#
q_dl_vp_lb_list   = x_q_dl_vp.min(axis = 0) #(3653,)
q_dl_vp_ub_list   = x_q_dl_vp.max(axis = 0) #(3653,)

#**************************************************************************************************;
#  6b. Plot best DL, GLUE, and DDS time-series with confidence interval (calib and valid periods)  ;
#**************************************************************************************************;
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/GLUE_vs_Obs_q_cp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (GLUE)']
num_ticks      = 100
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'red', 'red']
#
plot_obs_vs_swat_data(cp_t_list, x_q_glue_cp[640,:], \
						q_glue_cp_lb_list, q_glue_cp_ub_list, x_q_obs_cp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list)  
#
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/GLUE_vs_Obs_q_vp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (GLUE)']
num_ticks      = 500
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'red', 'red']
#
plot_obs_vs_swat_data(vp_t_list, x_q_glue_vp[640,:], \
						q_glue_vp_lb_list, q_glue_vp_ub_list, x_q_obs_vp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list)
#
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/DDS_vs_Obs_q_cp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (DDS)']
num_ticks      = 100
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'green', 'green']
#
plot_obs_vs_swat_data(cp_t_list, x_q_dds_cp[229,:], \
						q_dds_cp_lb_list, q_dds_cp_ub_list, x_q_obs_cp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list)  
#
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/DDS_vs_Obs_q_vp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (DDS)']
num_ticks      = 500
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'green', 'green']
#
plot_obs_vs_swat_data(vp_t_list, x_q_dds_vp[4621,:], \
						q_dds_vp_lb_list, q_dds_vp_ub_list, x_q_obs_vp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list)
#
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/CNN_vs_Obs_q_cp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (CNN)']
num_ticks      = 100
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'blue', 'blue']
#
plot_obs_vs_swat_data(cp_t_list, x_q_dl_cp[3950,:], \
						q_dl_cp_lb_list, q_dl_cp_ub_list, x_q_obs_cp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list) 
#
str_x_label    = 'Time [Days]'
str_y_label    = r'Streamflow [$\frac{m^3}{s}$]'
fl_name        = path + 'Plots/SWAT_vs_Obs/CNN_vs_Obs_q_vp' 
legend_loc     = 'upper left'
obs_label_list = ['Observed', 'Calibrated SWAT (CNN)']
num_ticks      = 500
line_width_ens = [1.0, 1.0]
alpha_val      = 0.2
col_list       = ['black', 'blue', 'blue']
#
plot_obs_vs_swat_data(vp_t_list, x_q_dl_vp[2881,:], \
						q_dl_vp_lb_list, q_dl_vp_ub_list, x_q_obs_vp, \
						str_x_label, str_y_label, fl_name, \
						legend_loc, obs_label_list, num_ticks, \
						line_width_ens, alpha_val, col_list)

#**********************************************************************************;
#  6c. Plot best DL, GLUE, and DDS flow duration curves (calib and valid periods)  ;
#**********************************************************************************;
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_GLUE_vs_Obs_q_cp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (GLUE)', 'All GLUE sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.02
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'red', 'red']
#
plot_fdc_lines(exceedence_cp*100, q_glue_cp_fdc, q_glue_best_cp_fdc, q_obs_cp_fdc, \
                str_x_label, str_y_label, fig_name, num_glue_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_GLUE_vs_Obs_q_vp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (GLUE)', 'All GLUE sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.02
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'red', 'red']
#
plot_fdc_lines(exceedence_vp*100, q_glue_vp_fdc, q_glue_best_vp_fdc, q_obs_vp_fdc, \
                str_x_label, str_y_label, fig_name, num_glue_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_DDS_vs_Obs_q_cp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (DDS)', 'All DDS sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.02
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'green', 'green']
#
plot_fdc_lines(exceedence_cp*100, q_dds_cp_fdc, q_dds_best_cp_fdc, q_obs_cp_fdc, \
                str_x_label, str_y_label, fig_name, num_dds_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_DDS_vs_Obs_q_vp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (DDS)', 'All DDS sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.02
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'green', 'green']
#
plot_fdc_lines(exceedence_vp*100, q_dds_vp_fdc, q_dds_best_vp_fdc, q_obs_vp_fdc, \
                str_x_label, str_y_label, fig_name, num_dds_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_CNN_vs_Obs_q_cp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (CNN)', 'All CNN sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.002
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'blue', 'blue']
#
plot_fdc_lines(exceedence_cp*100, q_dl_cp_fdc, q_dl_best_cp_fdc, q_obs_cp_fdc, \
                str_x_label, str_y_label, fig_name, num_dl_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/SWAT_vs_Obs/FDC_CNN_vs_Obs_q_vp' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Best calibration set (CNN)', 'All CNN sets']
num_ticks       = 10
line_width_ens  = [1.0, 2.0, 0.12]
alpha_val       = 0.002
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'blue', 'blue']
#
plot_fdc_lines(exceedence_vp*100, q_dl_vp_fdc, q_dl_best_vp_fdc, q_obs_vp_fdc, \
                str_x_label, str_y_label, fig_name, num_dl_realz, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)

#**************************************************************************;
#  7. Plot size of mean variation and its probability (glue, dds, and dl)  ;
#**************************************************************************;
q_glue_mv_ctp   = 31.43770912408759 #Size of mean variation for calibration time period (glue)
q_dds_mv_ctp    = 26.301152756386863 #Size of mean variation for calibration time period (dds)
q_dl_mv_ctp     = 14.590690959279197 #Size of mean variation for calibration time period (dl)
#
q_glue_prob_ctp = 0.9826642335766423 #Prob that obs. data is contained within the lb and ub boundary (glue)
q_dds_prob_ctp  = 0.9872262773722628 #Prob that obs. data is contained within the lb and ub boundary (dl)
q_dl_prob_ctp   = 0.9680656934306568 #Prob that obs. data is contained within the lb and ub boundary (glue)
#
q_glue_mv_vtp   = 31.199448353134407 #Size of mean variation for validation time period (glue)
q_dds_mv_vtp    = 25.496431042417193 #Size of mean variation for validation time period (dds)
q_dl_mv_vtp     = 14.03329540719956  #Size of mean variation for validation time period (dl)
#
q_glue_prob_vtp = 0.993430057486997 #Prob that obs. data is contained within the lb and ub boundary (glue)
q_dds_prob_vtp  = 0.9920613194634547 #Prob that obs. data is contained within the lb and ub boundary (dds)
q_dl_prob_vtp   = 0.9937038050917054 #Prob that obs. data is contained within the lb and ub boundary (dl)
#
str_x_label     = 'Time period'
str_y_label     = 'Mean variation of streamflow' 
data_list       = ['Calibration', 'Validation']
x_pos_list      = [[0, 1], [0.15, 1.15], [0.3, 1.3]]
xticks_list     = [x_pos_list[1][0], x_pos_list[1][1]]
y_mv_list       = [[q_dl_mv_ctp, q_dl_mv_vtp],
                    [q_dds_mv_ctp,  q_dds_mv_vtp],
                    [q_glue_mv_ctp, q_glue_mv_vtp]]
ymin            = 0
ymax            = 35
width_val       = 0.15
loc_pos         = 'upper center'
fig_name        = path + 'Plots/SWAT_vs_Obs/Size_MV_GLUE_DDS_CNN'
plot_mv_prob_metrics(x_pos_list, y_mv_list, xticks_list, data_list, \
    ymin, ymax, loc_pos, str_x_label, str_y_label, fig_name)
#
str_x_label     = 'Time period'
str_y_label     = 'Probability' 
data_list       = ['Calibration', 'Validation']
x_pos_list      = [[0, 1], [0.15, 1.15], [0.3, 1.3]]
xticks_list     = [x_pos_list[1][0], x_pos_list[1][1]]
y_prob_list     = [[q_dl_prob_ctp, q_dl_prob_vtp],
                    [q_dds_prob_ctp,  q_dds_prob_vtp],
                    [q_glue_prob_ctp, q_glue_prob_vtp]]
ymin            = 0.9
ymax            = 1.0
width_val       = 0.15
loc_pos         = 'upper center'
fig_name        = path + 'Plots/SWAT_vs_Obs/Size_Prob_GLUE_DDS_CNN'
plot_mv_prob_metrics(x_pos_list, y_prob_list, xticks_list, data_list, \
    ymin, ymax, loc_pos, str_x_label, str_y_label, fig_name)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)