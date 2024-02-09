'''
Decomp/LoS
'''

import json
import sys
import os
import argparse
import pandas as pd
import pickle
import numpy as np
from random import shuffle
from scipy.special import softmax
from sklearn.metrics import average_precision_score,auc,roc_curve,precision_recall_curve



def get_args():
	parser = argparse.ArgumentParser()

	#log
	parser.add_argument("--log_file", type=str, help="log training status per epoch")
	parser.add_argument("--summary_file", type=str, help="performance summary")
	parser.add_argument("--model_name", type=str, help="name of saved model")

	#training params
	parser.add_argument("--n_repeat", type=int, default=5, help="repeat training n times")
	parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--max_epoch", type=int, default=100, help="max number of training epochs")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size")
	parser.add_argument("--early_stop", type=int, default=10, help="early stop steps")
	parser.add_argument("--pos_weight", type=float, default=1.0, help="weight for the positive class when neg weight = 1.0")
	parser.add_argument("--label_hours", type = int, default = 48, help = "x hours ahead predictions")

	#model params
	parser.add_argument("--rnn_layers", type=int, default=3, help="rnn layers")
	parser.add_argument("--rnn_h_dim", type=int, default=128, help="rnn hidden size")
	parser.add_argument("--z_dim", type=int, default=8, help="vae z dim")
	parser.add_argument("--L2_reg", type = float, default = 0.0, help = "L2 regularization of classifier weights")
	parser.add_argument("--dropout", type = float, default = 0.0, help = "dropout")

	parser.add_argument("--reconloss_decay_init", type = float, default = 1.0, help = "recon loss starting multiplier")
	parser.add_argument("--reconloss_decay_factor", type = float, default = 0.9, help = "recon loss decay ratio")
	return vars(parser.parse_args())


def writelog(ctnt, logfile):
	with open(logfile,'a+') as f:
		f.writelines(ctnt+'\n')

#y label, int
#N lengths, int
def get_labels(y:int, N:int, last_x_hours:int):
	if N<=last_x_hours:
		return [y]*N
	return [0]*(N-last_x_hours)+ [y]*last_x_hours

'''
====================
Data Pipe
====================
'''


def pad_one_stay(x, y, pad_len):
	N = len(x)
	if N<pad_len:
		n_feat = x.shape[-1]
		x = np.concatenate([x, np.zeros((pad_len-N, n_feat))])
		y = np.concatenate([y, np.zeros(pad_len-N)])

	return x,y

def generate_ts(X, y, seq_len):
	X_padded = []
	y_padded = []
	pad_len = max(seq_len)
	for Xi, yi in  zip(X, y):
		Xi, yi = pad_one_stay(Xi, yi, pad_len)
		X_padded.append(Xi)
		y_padded.append(yi)

	return np.stack(X_padded), np.stack(y_padded)


def generate_padded_batches(batch_size, x, y, seq_len, group_by_lengths = False, n_bins = 5):

	if group_by_lengths:
		data = list(zip(x, y, seq_len))
		N_data = len(x)
		lengths = [len(x[i]) for i in range(N_data)]
		inds = list(pd.qcut(lengths, n_bins, labels=False))

		#initialize pointers
		ptrs = [0]*n_bins
		for i in range(1,n_bins):
			ptrs[i] = ptrs[i-1] + inds.count(i-1)

		#rearrange data
		new_z = [0]*N_data
		for i in range(N_data):
			bin_idx = inds[i]
			new_z[ptrs[bin_idx]] = data[i]
			ptrs[bin_idx] += 1

		x, y, seq_len = zip(*new_z)
		del data
		del new_z

	batches = []
	begin = 0

	while begin < len(x):
		end = min(begin+batch_size, len(x))
		assert(end-begin>0)
		seq_len_slice = seq_len[begin:end]
		x_slice, y_slice = generate_ts(x[begin:end], y[begin:end], seq_len_slice)

		batches.append((x_slice, y_slice, seq_len_slice))
		begin += batch_size

	return batches


# ======= Metrics ===========#

def getMetrics(probas, y_true):
	#y_true = np.concatenate(y_true)
	#probas = np.concatenate(probas)
	#aucpr = average_precision_score(y_true, probas)
	probas = np.clip(np.nan_to_num(np.array(probas)),0.0,1.0)
	(precisions, recalls, thresholds) = precision_recall_curve(y_true, probas)
	aucpr = auc(recalls, precisions)
	fpr, tpr, thresh = roc_curve(y_true, probas)
	aucroc = auc(fpr, tpr)
	return aucpr, aucroc

def getThreshold_AUROC(probas, y_true):
	#y_true = np.concatenate(y_true)
	#probas = np.concatenate(probas)
	#aucpr = average_precision_score(y_true, probas)
	probas = np.clip(np.nan_to_num(np.array(probas)),0.0,1.0)
	(precisions, recalls, thresholds) = precision_recall_curve(y_true, probas)
	aucpr = auc(recalls, precisions)
	fpr, tpr, thresh = roc_curve(y_true, probas)
	J = tpr - fpr
	idx = argmax(J)
	best_thresh = thresholds[idx]
	return best_thresh
