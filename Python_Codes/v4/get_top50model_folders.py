#Get Top-50 model folders (standard scaler from 18750 CNN architectures)

import os
import time
import subprocess

#********************************************;
#  1. Set paths and top-50 DL model numbers  ;
#********************************************;
path    = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
bm_list = [46, 51, 52, 53, 56, 57, 58, 59, 196, \
			201, 203, 204, 206, 208, 356, 357, 358, \
			507, 657, 807, 3801, 3806, 3807, 3808, 5136, \
			7740, 8281, 8737, 8886, 8899, 8905, 8909, 9659, \
			9936, 10398, 12359, 13092, 13097, 13248, 13253, 13403, \
			13409, 13555, 14009, 14748, 15503, 15651, 16402, 17007, 17896]

if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
	tic = time.perf_counter()

	#***********************************;
	#  2. Copy top-50 DL model numbers  ;
	#***********************************;
	for i in bm_list:
		print(path + "1_InvCNNModel_ss/" + str(i) + "_model")
		cmd = "cp -R " + path + "1_InvCNNModel_ss/" + str(i) + "_model " + \
				path + "Top_50models_ss/"
		print(cmd)
		subprocess.call(cmd, shell=True)

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)