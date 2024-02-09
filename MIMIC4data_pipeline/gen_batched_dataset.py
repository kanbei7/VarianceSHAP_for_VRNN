import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import math
from scipy.signal import detrend
from scipy import stats
from tqdm import tqdm


datadir='datasets'
datafile='transformedB_data_f20_basic_none_multi_forward_binarymask.pkl'
N_FEAT=10
TIME_BATCH_SIZE=int(sys.argv[1])
MIN_LEN=48
N_FEAT_OUT=80
outname='transformedB_data_f%d_ffill_%02dhr_Minlen%02d'%(N_FEAT_OUT,TIME_BATCH_SIZE,MIN_LEN)
outname=os.path.join(datadir,outname)

with open(os.path.join(datadir,datafile),'rb') as f:
	X,Y,IDs=pk.load(f)

newX,newY,newIDs=[],[],[]
for i in range(len(X)):
	len_x = len(X[i])
	if len_x<MIN_LEN:
		continue
	if np.sum(X[i][:,N_FEAT:])<((len_x/TIME_BATCH_SIZE)*N_FEAT):
		continue

	newX.append(X[i])
	newY.append(Y[i])
	newIDs.append(IDs[i])

print('Num. of Stays: %d'%len(newX))
del X
del Y
del IDs

def fit_line(x,y):
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)	
	return slope, r_value**2

def tsFeat(data,m):
	if np.sum(m)>1:
		x,y=[],[]
		for i in range(len(data)):
			if m[i]>0:
				x.append(i)
				y.append(data[i])
				

		if np.sum(m)==2:
			assert(len(x)==2)
			return [y[0],y[-1],np.median(data), np.min(data), np.max(data), (y[1]-y[0])/(x[1]-x[0]), 0.0]

		if np.sum(m)==1:
			return [data[0],data[-1],y[0], y[0], y[0], np.nan, np.nan]

		slope, detrend_var = fit_line(x,y)
		return [y[0],y[-1],np.median(data), np.min(data), np.max(data), slope, detrend_var]


	else:
		return [data[0],data[-1],data[0], data[0], data[0],np.nan,np.nan]
	return [data[0],data[-1],data[0], data[0], data[0],np.nan,np.nan]



def createVec_per_tb(ts):
	res=[]
	for i in range(N_FEAT):
		res.extend(tsFeat(ts[:,i], ts[:,i+N_FEAT]))
	return res+[np.mean(ts[:,i+N_FEAT]) for i in range(N_FEAT)]

def createVec(X):
	seqlen=len(X)
	vec = np.array([createVec_per_tb(X[i*TIME_BATCH_SIZE:min((i+1)*TIME_BATCH_SIZE, seqlen),:]) for i in range(math.ceil(seqlen/TIME_BATCH_SIZE))])
	#ffill
	vec=pd.DataFrame(vec)
	vec.ffill(inplace=True)
	vec.fillna(0, inplace=True)
	return vec.values

newSeq_len=[]
for i in tqdm(range(len(newX))):
	newSeq_len.append(len(newX[i]))
	newX[i] = createVec(newX[i])


with open(outname,'wb') as f:
	pk.dump([newX,newY,newIDs,newSeq_len],f)