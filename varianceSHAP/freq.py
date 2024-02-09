import torch
import torch.nn as nn
import numpy as np
import os
import pickle as pk
import random
import sys
import time
from utils import get_labels,writelog,generate_padded_batches,getMetrics,getThreshold_AUROC
import math
from scipy.special import expit, logit
from sklearn.metrics import confusion_matrix
from vrnn_model import VRNN2
from random import shuffle
from tqdm import tqdm
from torch.autograd import grad,Variable
import shap
import pandas as pd
DATA_PREFIX='f30_'

#wrapper
class variance_wrapper_full(nn.Module):
	def __init__(self, model):
		super(variance_wrapper_full,self).__init__()
		self.model=model

	def get_derivative(self,z):
		z=Variable(z,requires_grad=True)
		y=self.model.clf(z)
		y_z=grad(y[:,0],z,grad_outputs=torch.ones_like(y[:,0]),create_graph=True)[0]
		return (y_z)**2
		
	def forward(self,x):
		res=self.model(x)
		#print(x.shape)
		z1,z2=res['mus'][:,-1,:],res['logvar'][:,-1,:]
		y_z=self.get_derivative(z1)
		
		total_var=torch.bmm(z2.exp().unsqueeze(1),y_z.unsqueeze(2))
		#print(total_var.squeeze(1).shape)
		return total_var.squeeze(1)
		



#---------- parse args --------------#

DATA_DIR = '/mnt/hdd/mimic4-benchmark/decomp/data'
model_name = 'vrnn2-2_00.pt'
model_path = os.path.join('models',model_name+'.ckpt')
lr = 1e-3
batch_size = 16
dropout = 0.0
rnn_layers = 2
h_dim = 128
z_dim =128
wd = 0.0
label_hours = 48

with open(os.path.join(DATA_DIR,DATA_PREFIX+'train.pkl'),'rb') as f:
	train_X, train_y,train_id = pk.load(f)
assert(len(train_X)==len(train_y))
idx_X1=[i for i in range(len(train_X)) if 48<=len(train_X[i])<=61]
idx_X2=[i for i in range(len(train_X)) if 61<len(train_X[i])<=82]
idx_X3=[i for i in range(len(train_X)) if 82<len(train_X[i])<=124]
idx_X4=[i for i in range(len(train_X)) if 124<len(train_X[i])]
idx=idx_X1+idx_X2+idx_X3+idx_X4
train_X=[train_X[i] for i in range(len(train_X)) if i in idx]
train_y = [train_y[i] for i in range(len(train_id)) if i in idx]
print(pd.Series([len(x) for x in train_X]).describe())

train_y = [ get_labels(y, len(x), label_hours) for x,y in zip(train_X, train_y)]
train_seq_lengths = [len(train_X[i]) for i in range(len(train_X))]

train_batches = generate_padded_batches(batch_size, train_X, train_y, train_seq_lengths)
train_dataloader = [(i, torch.from_numpy(train_batches[i][0]).float(), torch.from_numpy(train_batches[i][1]).float(), train_batches[i][2])
					for i in range(len(train_batches))]



with open(os.path.join(DATA_DIR,DATA_PREFIX+'test.pkl'),'rb') as f:
	test_X, test_y,testid = pk.load(f)

idx_X1=[i for i in range(len(test_X)) if 48<=len(test_X[i])<=61]
idx_X2=[i for i in range(len(test_X)) if 61<len(test_X[i])<=82]
idx_X3=[i for i in range(len(test_X)) if 82<len(test_X[i])<=124]
idx_X4=[i for i in range(len(test_X)) if 124<len(test_X[i])]
idx=idx_X1+idx_X2+idx_X3+idx_X4
test_X=[test_X[i] for i in range(len(testid)) if i in idx]
test_y = [test_y[i] for i in range(len(testid)) if i in idx]
print(pd.Series([len(x) for x in test_X]).describe())


test_y = [ get_labels(y, len(x), label_hours) for x,y in zip(test_X, test_y)]
test_seq_lengths = [len(test_X[i]) for i in range(len(test_X))]
test_batches = generate_padded_batches(batch_size, test_X, test_y,  test_seq_lengths)
test_dataloader = [(i, torch.from_numpy(test_batches[i][0]).float(), torch.from_numpy(test_batches[i][1]).float(), test_batches[i][2])
					for i in range(len(test_batches))]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model = VRNN2(
	input_dim=30, 
	rnn_h_dim=h_dim, 
	rnn_n_layers=rnn_layers, 
	z_dim=z_dim
	)

model.load_state_dict(torch.load(os.path.join('models', model_name)))
model.to(device)

res=[]
#wrapper
for t in tqdm(range(120,24,-1)):
	idx_train=[i for i in range(len(train_X)) if len(train_X[i])>=t]
	idx_test=[i for i in range(len(test_X)) if len(test_X[i])>=t]
	idx_background = np.random.choice(idx_train,min(len(idx_train),6000),replace=False)
	idx_shap = np.random.choice(idx_test,min(len(idx_test),1000),replace=False)

	bk_data=[torch.from_numpy(train_X[i][:t,:]).float() for i in range(len(train_X)) if i in idx_background]
	test_data=[torch.from_numpy(test_X[i][:t,:]).float() for i in range(len(test_X)) if i in idx_shap]
	bk_data=torch.stack(bk_data)
	test_data=torch.stack(test_data)
	vwrp= variance_wrapper_full(model)
	e = shap.GradientExplainer(vwrp, [bk_data])
	svz = e.shap_values([test_data])
	tmp=[torch.stack([torch.from_numpy(sv[-1]).float() for sv in svz]), test_data[:,-1,:]]
	
	res.append(svz)

	with open('res%d.pkl'%t,'wb') as f:
		pk.dump(tmp,f)
with open('allres.pkl','wb') as f:
	pk.dump(res,f)
