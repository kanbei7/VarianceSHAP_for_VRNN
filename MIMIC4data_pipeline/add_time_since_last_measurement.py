import numpy as np
import os
import pickle as pk
import math
from tqdm import tqdm

def calc_log_time(m):
	prev_one=-1
	ans=[1]*len(m)
	for i in range(len(m)):
		if m[i]>0:
			prev_one=i
			ans[i]=0
		else:
			ans[i]=math.log(i+1-prev_one,24)
	return ans


fname_f20 = 'transformedB_data_f20_basic_none_multi_forward_binarymask.pkl'
fname_all = 'transformedB_data_all_basic_none_multi_forward_binarymask.pkl'
DATA_DIR = 'datasets'

with open(os.path.join(DATA_DIR,fname_f20),'rb') as f:
	data_X, data_y, sid = pk.load(f)

new_dataX=[]
#for i in tqdm(range(1000)):
for i in tqdm(range(len(data_X))):
	x=data_X[i]

	tm=[]
	for j in range(10,20):
		tm.append(calc_log_time(x[:,j]))
	tm=np.stack(tm,axis=1)
	new_x=np.concatenate([x,tm],axis=1)
	new_dataX.append(new_x)


with open(os.path.join(DATA_DIR,'v2_'+fname_f20),'wb') as f:
	pk.dump([new_dataX, data_y, sid],f)



mask_idx=np.arange(42,95)


with open(os.path.join(DATA_DIR,fname_all),'rb') as f:
	data_X, data_y, sid = pk.load(f)

new_dataX=[]
#for i in tqdm(range(1000)):
for i in tqdm(range(len(data_X))):
	x=data_X[i]

	tm=[]
	for j in mask_idx:
		tm.append(calc_log_time(x[:,j]))
	tm=np.stack(tm,axis=1)
	new_x=np.concatenate([x,tm],axis=1)
	new_dataX.append(new_x)


with open(os.path.join(DATA_DIR,'v2_'+fname_all),'wb') as f:
	pk.dump([new_dataX, data_y, sid],f)

