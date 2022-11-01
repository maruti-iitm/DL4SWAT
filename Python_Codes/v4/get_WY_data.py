# Stream discharge data subset for DL and SWAT model calibration (SWAT_v5 -- 1000 sobol sequence with only one soil layer)
#   SWAT simulation time-stamp: 2011/1/1 to 12/31/2016 (2192 time-stamps)
# DL and SWAT model calibration:
#   Water Year (WY) from 2014 to 2016
#   Oct-1-2013 to Sep-30-2016
#   Create different .csv files for this data subset (params and flow)
#
#   https://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
#   https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
#   https://www.timeanddate.com/date/durationresult.html?m1=1&d1=1&y1=2000&m2=12&d2=27&y2=2019 #7300 days
# AUTHOR: Maruti Kumar Mudunuru

import time
import datetime
import pandas as pd
from datetime import date, timedelta
import numpy as np

np.set_printoptions(precision=2)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path      = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs  = path + 'Data/Obs_Data/' #Obs data
path_ind  = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
#
df_p      = pd.read_csv(path_swat + 'parameter_ensemble_SWAT_20_no-soil_Sobol5000real.csv', \
                        index_col = 0) #5000 sobol realz (20 SWAT parameters) [5000 rows x 20 columns]
df_q      = pd.read_csv(path_swat + 'outflow_ARW_2011_2016_mi.csv', \
                        index_col = 0) #Daily discharge [2192 rows x 1000 columns]
df_q_obs  = pd.read_csv(path_obs + 'obs_flow_data.csv', index_col = 0) #[14885 rows x 1 columns]
#
print(np.argwhere(np.isnan(df_q.values)))

#*******************************************************************;
#  2. Get dates and save updated CSV file (2011/1/1 to 12/31/2016)  ;
#*******************************************************************;
sdate    = date(2011, 1, 1) # start date
edate    = date(2016, 12, 31) # end date
#
delta    = edate - sdate  # as timedelta
day_list = [] #days list
#
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    day_list.append(str(day))
    print(i,day)

df_q_copy = df_q.copy(deep=True)
df_q_copy.insert(0, "Days", day_list)
#
df_q_copy.to_csv(path_swat + 'flow_sobol_1000realz.csv') #[2192 rows x 5001 columns]

#**********************************************;
#  3. Save WY data and associated sobol realz  ;
#**********************************************;
df_p_copy   = df_p.iloc[0:1000,:].copy(deep=True) #First 1000 Sobol realz --> SWAT parameters [1000 rows x 20 columns]
df_q_wy     = df_q_copy.iloc[1004:2100,:].copy(deep=True) #WY-2014 to WY-2016
df_q_obs_wy = df_q_obs.iloc[12327:13423,:].copy(deep=True) #WY-2014 to WY-2016
#
df_p_copy.to_csv(path_swat + 'swat_params_sobol_1000realz.csv') #[1000 rows x 20 columns]
df_q_wy.to_csv(path_swat + 'flow_wy_sobol_1000realz.csv') #[1096 rows x 1001 columns] WY-2014 to WY-2016
df_q_obs_wy.to_csv(path_obs + 'obs_flow_wy_data.csv') #[1096 rows x 1001 columns] WY-2014 to WY-2016

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)