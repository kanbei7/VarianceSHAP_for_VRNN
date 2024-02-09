import pandas as pd
import numpy as np
import sys
import os
import pickle as pk
from sklearn import preprocessing
from tqdm import tqdm

inputdata='data_f20_basic_none_multi_forward_binarymask.pkl'

FEAT_NAMES=[
	'abpdia',
	'abpmean',
	'abpsys',
	'fio2',
	'glucose',
	'hr',
	'ph',
	'rr',
	'spo2',
	'temp'
]

N_feat=10

def fix(ts):
	#SpO2
	ts[:,8]=100-ts[:,8]
	#Glucose
	ts[:,4]=np.clip(ts[:,4],a_min=None,a_max=600)
	#ph
	ts[:,6][ts[:,6]<=0]=7.39

	#RR
	#ts[:,7]=np.clip(ts[:,7],a_min=None,a_max=1000)
	ts[:,7][ts[:,7]>1000]=1000
	#temp
	ts[:,9][ts[:,9]<=0]=37

	return ts


with open(os.path.join('datasets',inputdata),'rb') as f:
	data = pk.load(f)



X = [fix(t[1]) for t in data]
Y = [t[0] for t in data]
IDs = [t[2] for t in data]


with open(os.path.join('datasets','fixed_'+inputdata),'wb') as f:
	pk.dump([X,Y, IDs],f)



feat=[[] for _ in range(N_feat)]
for ts in tqdm(X):
	m=ts[:,N_feat:]
	for i in range(N_feat):
		v=[a for a,b in zip(ts[:,i],m[:,i]) if b>0]
		feat[i].extend(v)


for i in range(10):
	print(FEAT_NAMES[i])
	print(pd.Series(feat[i]).describe())
	feat[i]=np.array(feat[i]).reshape(-1,1)
	print('\n')



'''
template_A=['z']*N_feat
template_A[4]='p'
template_A[8]='p'
'''

template_B=['p']*N_feat
template_B[9]='z'
template_B[6]='z'

def learn_trans_func(feat,trans_template):
	trans_func=[]
	for i in range(N_feat):
		if trans_template[i]=='p':
			scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
		else:
			scaler = preprocessing.StandardScaler()
		scaler.fit(feat[i])
		trans_func.append(scaler)

	return trans_func


def transform_data(X,trans_func):
	for ts in tqdm(X):
		for i in range(N_feat):
			ts[:,i]=trans_func[i].transform(ts[:,i].reshape(-1,1)).reshape(-1)
	return X
'''
trans_func=learn_trans_func(feat,template_A)
newX=transform_data(X,trans_func)
with open(os.path.join('datasets','data_f20_basic_transformedA_multi_forward_binarymask.pkl'),'wb') as f:
	pk.dump([X,Y],f)

'''
trans_func=learn_trans_func(feat,template_B)
newX=transform_data(X,trans_func)
assert(len(Y)==len(IDs))
with open(os.path.join('datasets','transformedB_'+inputdata),'wb') as f:
	pk.dump([X,Y, IDs],f)



#check stat after transform
newfeat=[[] for _ in range(N_feat)]
for ts in tqdm(newX):
	m=ts[:,N_feat:]
	for i in range(N_feat):
		v=[a for a,b in zip(ts[:,i],m[:,i]) if b>0]
		newfeat[i].extend(v)

for i in range(N_feat):
	print(FEAT_NAMES[i])
	print(pd.Series(newfeat[i]).describe())
	newfeat[i]=np.array(newfeat[i])
	print('\n')