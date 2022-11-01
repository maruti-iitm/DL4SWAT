# Stream discharge data subset -- Observational data
#   WY-1999 to WY-2013 is the validation
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
df_q_obs  = pd.read_csv(path_obs + 'obs_flow_data.csv', index_col = 0) #[14885 rows x 1 columns]
#
print(np.argwhere(np.isnan(df_q_obs.values)))

#******************************;
#  2. Save WY validation data  ;
#******************************;
df_q_obs_wy = df_q_obs.iloc[7213:12327,:].copy(deep=True) #WY-1999 to WY-2013
#
df_q_obs_wy.to_csv(path_obs + 'obs_flow_wyval_data_1.csv') #[5114 rows x 1 columns] WY-1999 to WY-2013

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)