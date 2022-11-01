import glob
import time

#============================;
#  1. Start processing time  ;
#============================;
tic = time.perf_counter()

path = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
#path = "/Volumes/OneTouchMKM/2_DL_SWAT/"
hp_h5_list = glob.glob(path + '1_InvCNNModel_ss/**/*.h5', recursive = True)
print(len(hp_h5_list))

#=========================;
#  2.End processing time  ;
#=========================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)