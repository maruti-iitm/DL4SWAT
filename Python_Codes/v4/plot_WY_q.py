# Create data plots of stream discharge and flow duration curves --- SWAT ensembles vs. obs data
#   1000 Sobol's sequence
#   Water Year (WY) from 2014 to 2016 (see process_q_ET_1oyrs_plots.py)
#   Oct-1-2013 to Sep-30-2016
# FLOW DURATION CURVE
#	https://stackoverflow.com/questions/49304516/plotting-a-flow-duration-curve-for-a-range-of-several-timeseries-in-python/49304517
# AUTHOR: Maruti Kumar Mudunuru

import os
import pickle
import time
import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.set_printoptions(precision=2)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#===========================================;
#  Function-1: Plot obs vs. SWAT ensembles  ;
#===========================================;
def plot_obs_vs_swat_data(num_realz, t_list, y_ens, y_mean, y_lb, y_ub, y_obs, \
                          str_x_label, str_y_label, fig_name, \
                          legend_loc, plot_label_list, num_ticks, \
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
                 marker = None, label = plot_label_list[0]) #Obs. discharge data
    sns.lineplot(t_list, y_mean, linestyle = 'dashed', linewidth = line_width_ens[1], color = col_list[1], \
                 marker = None, label = plot_label_list[1]) #Mean discharge data
    #ax.fill_between(t_list, y_lb, y_ub, linestyle = 'solid', linewidth = 0.5, \
    #                color = col_list[2], alpha = alpha_val) #Mean +/ 2*std or 95% CI
    for i in range(0,num_realz):
        #print(i)
        if i == 0:
            sns.lineplot(t_list, y_ens[:,i], linestyle = 'solid', \
                        linewidth = 0.1, color = col_list[2], \
                        marker = None, label = plot_label_list[2], \
                        alpha = alpha_val) #SWAT ensemble discharge FDC
        else:
            sns.lineplot(t_list, y_ens[:,i], linestyle = 'solid', \
                        linewidth = 0.1, color = col_list[2], \
                        marker = None, alpha = alpha_val) #SWAT ensemble discharge FDC
    tick_spacing = num_ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = legend_loc)
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

#=================================================================;
#  Function-2: Plot flow duration curve (obs vs. SWAT ensembles)  ;
#=================================================================;
def plot_flow_duration_curves(x_list, y_mean, y_lb, y_ub, y_obs, \
                                str_x_label, str_y_label, fig_name, \
                                legend_loc, plot_label_list, num_ticks, \
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
    ax.plot(x_list, y_obs, linestyle = 'solid', linewidth = line_width_ens[0], color = col_list[0], \
                 marker = None, label = plot_label_list[0]) #Obs. discharge FDC
    ax.plot(x_list, y_mean, linestyle = 'dashed', linewidth = line_width_ens[1], color = col_list[1], \
                 marker = None, label = plot_label_list[1]) #Mean discharge FDC
    ax.fill_between(x_list, y_lb, y_ub, linestyle = 'solid', linewidth = 0.5, \
                    color = col_list[2], alpha = alpha_val) #Mean +/ 2*std or 95% CI
    ax.set_yscale('log')
    tick_spacing = num_ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = legend_loc)
    #ax.set_aspect(1./ax.get_data_ratio())
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

#=================================================================;
#  Function-3: Plot flow duration curve (obs vs. SWAT ensembles)  ;
#=================================================================;
def plot_fdc_lines(x_list, y_ens, y_mean, y_obs, str_x_label, str_y_label, fig_name, \
                    legend_loc, plot_label_list, num_ticks, \
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
                        linewidth = 0.5, color = col_list[2], \
                        marker = None, label = plot_label_list[2], \
                        alpha = alpha_val) #SWAT ensemble discharge FDC
        else:
            ax.plot(x_list, y_ens[:,i], linestyle = 'solid', \
                        linewidth = 0.5, color = col_list[2], \
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
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path       = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat  = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs   = path + 'Data/Obs_Data/' #Obs data
path_ind   = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
#
num_realz  = 1000 #Sobol realz
df_p       = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
                        index_col = 0) #[1000 rows x 20 columns]
df_q       = pd.read_csv(path_swat + 'flow_wy_sobol_1000realz.csv', \
                        index_col = 0) #[1096 rows x 1001 columns] WY-2014 to WY-2016
df_q_obs   = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016
#
q_obs      = df_q_obs.values[:,0] #(1096,)
q_mean     = df_q.iloc[:,1:].mean(axis = 1).values #Mean q for each time-step for 1000 sobol realz (1096,)
q_std      = df_q.iloc[:,1:].std(axis = 1).values #Std q for each time-step for 1000 sobol realz (1096,)
q_lb       = df_q.iloc[:,1:].min(axis = 1).values #Min q for each time-step for 1000 sobol realz (1096,)
q_ub       = df_q.iloc[:,1:].max(axis = 1).values #Max q for each time-step for 1000 sobol realz (1096,)
#
t_list     = df_q.iloc[:,0].to_list() #(1096,)
#
q_obs_fdc  = np.sort(q_obs)[::-1] #(1096,)
#
q_swat     = df_q.iloc[:,1:].values #(1096, 1000)
q_swat_fdc = np.sort(q_swat, axis = 0)[::-1] #(1096, 1000)
q_mean_fdc = np.sort(q_mean)[::-1] #(1096,)
q_lb_fdc   = np.min(q_swat_fdc, axis=1) #(1096,)
q_ub_fdc   = np.max(q_swat_fdc, axis=1) #(1096,)
#
exceedence = np.arange(1.0,len(q_obs_fdc)+1)/len(q_obs_fdc) #(1096,)

#********************************************;
#  2. Plot discharge data vs SWAT ensembles  ;
#********************************************;
str_x_label     = 'Time [Days]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/Discharge/Mean_vs_Obs_q_Sobol1000realz' 
legend_loc      = 'upper right'
plot_label_list = ['Observed', 'Mean ensemble', 'SWAT ensembles']
num_ticks       = 100
line_width_ens  = [2.5, 2.5]
alpha_val       = 0.05
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'rosybrown', 'gainsboro']
#
plot_obs_vs_swat_data(num_realz, t_list, q_swat, q_mean, q_lb, q_ub, q_obs, \
                        str_x_label, str_y_label, fig_name, \
                        legend_loc, plot_label_list, num_ticks, \
                        line_width_ens, alpha_val, col_list)

#*********************************************************;
#  3. Plot flow duration curves (Obs vs. SWAT ensembles)  ;
#*********************************************************;
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/Discharge/Mean_vs_Obs_qFDC1_Sobol1000realz' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Mean ensemble']
num_ticks       = 10
line_width_ens  = [1.0, 1.0]
alpha_val       = 0.2
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'rosybrown', 'gainsboro']
#
plot_flow_duration_curves(exceedence*100, q_mean_fdc, q_lb_fdc, q_ub_fdc, q_obs_fdc, \
                            str_x_label, str_y_label, fig_name, \
                            legend_loc, plot_label_list, num_ticks, \
                            line_width_ens, alpha_val, col_list)
#
str_x_label     = r'Flow exceedence [\%]'
str_y_label     = r'Streamflow [$\frac{m^3}{s}$]'
fig_name        = path + 'Plots/Discharge/Mean_vs_Obs_qFDC2_Sobol1000realz' 
legend_loc      = 'lower left'
plot_label_list = ['Observed', 'Mean ensemble', 'SWAT ensembles']
num_ticks       = 10
line_width_ens  = [1.0, 1.0]
alpha_val       = 0.2
#col_list        = ['black', 'rosybrown', 'bisque']
col_list        = ['black', 'rosybrown', 'gainsboro']
#
plot_fdc_lines(exceedence*100, q_swat_fdc, q_mean_fdc, q_obs_fdc, \
                str_x_label, str_y_label, fig_name, \
                legend_loc, plot_label_list, num_ticks, \
                line_width_ens, alpha_val, col_list)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)