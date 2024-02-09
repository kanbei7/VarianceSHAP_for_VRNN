import pandas as pd
import numpy as np
import sys
import os
import pickle as pk
from sklearn import preprocessing
from tqdm import tqdm

inputdata='data_all_basic_none_multi_forward_binarymask.pkl'
N_feat=42

with open(os.path.join('datasets', 'feature_names_all_basic_none_multi_forward_binarymask.txt'), 'r') as f:
	FEAT_NAMES=[l.strip() for l in f.readlines() if l.strip()!='']


name2idx={FEAT_NAMES[i]:i for i in range(len(FEAT_NAMES))}

spo2idx=name2idx['spo2']
tempidx=name2idx['temp']
rridx=name2idx['rr']
phbloodidx=name2idx['ph-blood']
phvenousidx=name2idx['ph-venous']
glucosebloodidx=name2idx['glucose-blood']
glucoseserumidx=name2idx['glucose-serum']

spo2_maskidx=name2idx['spo2_mask']
temp_maskidx=name2idx['temp_mask']
rr_maskidx=name2idx['rr_mask']
phblood_maskidx=name2idx['ph-blood_mask']
phvenous_maskidx=name2idx['ph-venous_mask']
glucoseblood_maskidx=name2idx['glucose-blood_mask']
glucoseserum_maskidx=name2idx['glucose-serum_mask']


def fix(ts):
	#SpO2
	ts[:,spo2idx]=100-ts[:,spo2idx]
	#Glucose
	ts[:,glucosebloodidx]=np.clip(ts[:,glucosebloodidx],a_min=None,a_max=1000)
	ts[:,glucoseserumidx]=np.clip(ts[:,glucoseserumidx],a_min=None,a_max=1000)
	#ph
	ts[:,phbloodidx][ts[:,phbloodidx]<=0]=7.39
	ts[:,phvenousidx][ts[:,phvenousidx]<=0]=7.39

	#RR
	ts[:,rridx][ts[:,rridx]>1000]=1000
	#temp
	ts[:,tempidx][ts[:,tempidx]<=0]=37

	return ts


with open(os.path.join('datasets',inputdata),'rb') as f:
	data = pk.load(f)



X = [fix(t[1]) for t in data]
Y = [t[0] for t in data]
IDs = [t[2] for t in data]


with open(os.path.join('datasets',inputdata),'wb') as f:
	pk.dump([X,Y, IDs],f)



feat=[[] for _ in range(N_feat)]

for ts in tqdm(X):
	for i in range(N_feat):	
		mi=name2idx[FEAT_NAMES[i]+'_mask'] 
		v=[a for a,b in zip(ts[:,i],ts[:,mi]) if b>0]
		feat[i].extend(v)



for i in range(N_feat):
	#print(FEAT_NAMES[i])
	#print(pd.Series(feat[i]).describe())
	feat[i]=np.array(feat[i]).reshape(-1,1)
	#print('\n')





template_B=['p']*N_feat
template_B[phbloodidx]='z'
template_B[phvenousidx]='z'
template_B[tempidx]='z'

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

trans_func=learn_trans_func(feat,template_B)
newX=transform_data(X,trans_func)
assert(len(Y)==len(IDs))
with open(os.path.join('datasets','transformedB_'+inputdata),'wb') as f:
	pk.dump([X,Y, IDs],f)



#check stat after transform
'''
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
'''