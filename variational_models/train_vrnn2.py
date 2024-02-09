'''
for linux
'''
import torch
import torch.nn as nn
import numpy as np
import os
import pickle as pk
import random
import sys
import time
import utils
import math
from scipy.special import expit, logit
from sklearn.metrics import confusion_matrix
from vrnn_model import VRNN2
from random import shuffle
from tqdm import tqdm



DATA_PREFIX='f30_'

#---------- parse args --------------#
args = utils.get_args()
DATA_DIR = '/mnt/hdd/mimic4-benchmark/decomp/data'
model_name = args['model_name']
model_path = os.path.join('models',model_name+'.ckpt')
N_REPEATS=args['n_repeat']
logfile = os.path.join('log',args['log_file'])
summaryfile = os.path.join('res',args['summary_file'])
lr = args['learning_rate']
n_epoch = args['max_epoch']
batch_size = args['batch_size']
max_earlystop = args['early_stop']
pos_weight  = args['pos_weight']
class_weight = torch.tensor([1.0, pos_weight])
pos_weight = torch.tensor([pos_weight])
dropout = args['dropout']
rnn_layers = args['rnn_layers']
h_dim = args['rnn_h_dim']
z_dim =args['z_dim']
wd = args['L2_reg']
label_hours = args['label_hours']
#reconloss_decay_init = args['reconloss_decay_init']
#reconloss_decay_factor = args['reconloss_decay_factor']

#---------- Config --------------#
writelog = utils.writelog
generate_padded_batches = utils.generate_padded_batches
getMetrics = utils.getMetrics
getThreshold_AUROC = utils.getThreshold_AUROC

# ---------- Read Data ---------- #
generate_padded_batches = utils.generate_padded_batches

with open(os.path.join(DATA_DIR,DATA_PREFIX+'train.pkl'),'rb') as f:
	train_X, train_y,_ = pk.load(f)
train_y = [ utils.get_labels(y, len(x), label_hours) for x,y in zip(train_X, train_y)]
train_seq_lengths = [len(train_X[i]) for i in range(len(train_X))]

with open(os.path.join(DATA_DIR,DATA_PREFIX+'val.pkl'),'rb') as f:
	eval_X, eval_y,_ = pk.load(f)
eval_y = [ utils.get_labels(y, len(x), label_hours) for x,y in zip(eval_X, eval_y)]
eval_seq_lengths = [len(eval_X[i]) for i in range(len(eval_X))]

with open(os.path.join(DATA_DIR,DATA_PREFIX+'test.pkl'),'rb') as f:
	test_X, test_y,_ = pk.load(f)
test_y = [ utils.get_labels(y, len(x), label_hours) for x,y in zip(test_X, test_y)]
test_seq_lengths = [len(test_X[i]) for i in range(len(test_X))]

x_n_feat = train_X[0].shape[1]
print("data loaded.")


#print(x_n_feat)
#generate batches
eval_batches = generate_padded_batches(batch_size, eval_X, eval_y,  eval_seq_lengths)
val_dataloader = [(i, torch.from_numpy(eval_batches[i][0]).float(), torch.from_numpy(eval_batches[i][1]).float(), eval_batches[i][2])
					for i in range(len(eval_batches))]

del eval_X
del eval_y 
del eval_seq_lengths
del eval_batches

test_batches = generate_padded_batches(batch_size, test_X, test_y,  test_seq_lengths)
test_dataloader = [(i, torch.from_numpy(test_batches[i][0]).float(), torch.from_numpy(test_batches[i][1]).float(), test_batches[i][2])
					for i in range(len(test_batches))]
del test_X
del test_y 
del test_seq_lengths
del test_batches

# ============= Data Loaded ========== #


# ---------- Start Training ---------- #
criterion = nn.BCEWithLogitsLoss(reduction='sum')
reconcriterion=nn.MSELoss(reduction='sum')

def generate_mask(lengths):
	batch_max_len=max(lengths)
	lengths=torch.Tensor(lengths).long()
	return (torch.arange(batch_max_len)[None, :] < lengths[:, None]).int()

def selectrecon_loss(y, lengths):
	y = torch.cat([y[i,:lengths[i]-1,:].view(-1,x_n_feat) for i in range(len(lengths))])
	return y.view(-1,1)

c_sqrt2=math.sqrt(2)
def standard_cdf(x):
	return 0.5*(1+torch.erf(x/c_sqrt2))

def select_loss(y, lengths):
	y = torch.cat([y[i,:lengths[i]].view(-1) for i in range(len(lengths))])
	return y

def train(epoch, model, dataloader):
	start_time = time.time()
	model.train()
	total_loss,pred_loss=0.0,0.0
	kldloss,nllloss=0.0,0.0
	total_y_scores,total_y_true = [],[]

	for batch_idx, X, y_true, lengths in dataloader:
		N = sum(lengths)
		mask=generate_mask(lengths).to(device)
		X,y_true = X.to(device),y_true.to(device)

		optimizer.zero_grad()
		res_dict = model(X, mask,is_train=True)

		x_recon = selectrecon_loss(res_dict['x_recon'], lengths)
		x_true = selectrecon_loss(X[:,1:,:], lengths)
		y_pred = select_loss((res_dict['y_score']), lengths)
		y_true = select_loss(y_true, lengths)

		#assert(N>0)
		reconloss=torch.sqrt(reconcriterion(standard_cdf(x_recon), standard_cdf(x_true)) / N)
		#loss = reconloss_decay_init*np.power(reconloss_decay_factor,epoch-1)*torch.sqrt(reconcriterion(standard_cdf(x_recon), standard_cdf(x_true)) / N) + criterion(y_pred, y_true) / N  + wd*res_dict['clf_weights']
		
		loss = (criterion(y_pred, y_true) + res_dict['kld_loss']+torch.clamp(res_dict['nll_loss'], min=-10.0, max=10.0))/N + reconloss + wd*model.weight_sum()
		loss.backward()
		optimizer.step()

		kldloss+=float((res_dict['kld_loss']/N).item())
		nllloss+=float((res_dict['nll_loss']/N).item())
		total_loss+=float(loss.item())
		total_y_scores.extend(expit(y_pred.cpu().detach().numpy()))
		total_y_true.extend(y_true.cpu().numpy())

	kldloss=kldloss/len(dataloader)
	nllloss=nllloss/len(dataloader)
	total_loss=total_loss/len(dataloader)
	current_aucpr, current_aucroc = getMetrics(total_y_scores, total_y_true)
	pred_loss = criterion(torch.Tensor(total_y_scores), torch.Tensor(total_y_true))/len(total_y_true)
	threshold_auroc = 0.5
	#writelog
	writelog("[Train Epoch %d]"%epoch+"[Time: %.2f min]"%((time.time()-start_time)/60.0)+
		"Total Loss: %.4f , AUCPR: %.4f , AUCROC: %.4f,CLFLoss: %.8f,KLDLoss: %.8f,NLLLoss: %.8f" %(total_loss, current_aucpr, current_aucroc,pred_loss,kldloss,nllloss),logfile)
	return threshold_auroc,nllloss

def eval(test_or_val, epoch, model, dataloader, last_best_aupr, last_auroc, threshold):
	start_time = time.time()
	model.eval()
	total_loss,pred_loss=0.0,0.0
	kldloss,nllloss=0.0,0.0
	total_y_scores,total_y_true = [],[]

	with torch.no_grad():
		for batch_idx, X, y_true,  lengths in dataloader:
			N = sum(lengths)
			mask=generate_mask(lengths).to(device)
			X,y_true = X.to(device),y_true.to(device)

			res_dict = model(X, mask)
			x_recon = selectrecon_loss(res_dict['x_recon'], lengths)
			x_true = selectrecon_loss(X[:,1:,:], lengths)
			y_pred = select_loss((res_dict['y_score']), lengths)
			y_true = select_loss(y_true, lengths)
			#assert(N>0)
			reconloss=torch.sqrt(reconcriterion(standard_cdf(x_recon), standard_cdf(x_true)) / N)
			loss = (criterion(y_pred, y_true) + res_dict['kld_loss']+torch.clamp(res_dict['nll_loss'], min=-10.0, max=10.0))/N + reconloss + wd*model.weight_sum()
			total_loss+=float(loss.item())
			total_y_scores.extend(expit(y_pred.cpu().detach().numpy()))
			total_y_true.extend(y_true.cpu().numpy())
			kldloss+=float((res_dict['kld_loss']/N).item())
			nllloss+=float((res_dict['nll_loss']/N).item())
	
	kldloss=kldloss/len(dataloader)
	nllloss=nllloss/len(dataloader)
	total_loss=total_loss/len(dataloader)
	current_aucpr, current_aucroc = getMetrics(total_y_scores, total_y_true)
	pred_loss = criterion(torch.Tensor(total_y_scores), torch.Tensor(total_y_true))/len(total_y_true)

	#validation
	if test_or_val == 'val':
		changed = False
		if current_aucpr > last_best_aupr+0.00001:
			changed = True
			last_best_aupr = current_aucpr
			last_auroc = current_aucroc
			#torch.save(model.state_dict(), model_path)
		writelog("[Val Epoch %d]"%epoch+"[Time: %.2f min]"%((time.time()-start_time)/60.0)+ "[Best Val AUPR: %.4f]"%last_best_aupr + "[Time: %.2f min]"%((time.time()-start_time)/60.0)+
			"Total Loss: %.4f , AUCPR: %.4f , AUCROC: %.4f,CLFLoss: %.8f,KLDLoss: %.8f,NLLLoss: %.8f" %(total_loss, current_aucpr, current_aucroc,pred_loss,kldloss,nllloss),logfile)

		return  last_best_aupr, changed, last_auroc

	#test
	total_y_pred = [int(t>=threshold) for t in total_y_scores]
	tn, fp, fn, tp = confusion_matrix(total_y_true, total_y_pred).ravel()
	writelog("[Test Epoch %d]"%epoch+ "[Test AUPR: %.4f]"%current_aucpr + "[Time: %.2f min]"%((time.time()-start_time)/60.0)+
		"Total Loss: %.4f , AUCROC: %.4f,CLFLoss: %.8f,KLDLoss: %.8f,NLLLoss: %.8f" %(total_loss, current_aucroc,pred_loss,kldloss,nllloss),logfile)

	return current_aucpr, current_aucroc, tn, fp, fn, tp	


if __name__ == "__main__":
	writelog("decom vgru2",logfile)
	for param in args:
		writelog(param + ':' + str(args[param]), logfile)
	finishflag=False
	for i_rpt in range(N_REPEATS):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if finishflag:
			break
		#device = torch.device("cpu")
		best_val_aupr = -1.0
		val_auroc = -1.0
		early_stopping = 0
		test_aupr = 0
		test_roc = 0
		test_tn = 0
		test_fn = 0
		test_tp = 0
		test_fp = 0
		#dec_thresh = []

		model = VRNN2(
			input_dim=x_n_feat, 
			rnn_h_dim=h_dim, 
			rnn_n_layers=rnn_layers, 
			z_dim=z_dim
			)
		model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr = lr)

		end_epoch = 0
		for epoch in  tqdm(range(1, n_epoch+1)):
			#prepare data
			data = list(zip(train_X, train_y, train_seq_lengths))
			shuffle(data)
			train_X, train_y, train_seq_lengths = zip(*data)
			del data

			train_batches = generate_padded_batches(batch_size, train_X, train_y, train_seq_lengths)
			train_dataloader = [(i, torch.from_numpy(train_batches[i][0]).float(), torch.from_numpy(train_batches[i][1]).float(), train_batches[i][2])
								for i in range(len(train_batches))]
			del train_batches
			#train
			threshold_auroc,nllloss = train(epoch, model, train_dataloader)
			#dec_thresh.append(threshold_auroc)
			#val
			best_val_aupr, changed, val_auroc = eval('val', epoch, model, val_dataloader, best_val_aupr, val_auroc, threshold_auroc)
			if np.isnan(nllloss):
				break

			if changed:
				early_stopping = 0
				#test
				current_aucpr, current_aucroc, tn, fp, fn, tp = eval('test', epoch, model, test_dataloader, best_val_aupr, val_auroc, threshold_auroc)
				test_tn=tn
				test_fn=fn
				test_tp=tp
				test_fp=fp
				test_aupr=current_aucpr
				test_roc=current_aucroc
				end_epoch = epoch
				if best_val_aupr>0.42 or test_aupr>=0.417:
					finishflag=True
					torch.save(model.state_dict(),  os.path.join('models', model_name+'_%02d.pt'%i_rpt))
			else:
				early_stopping += 1
				if early_stopping >= max_earlystop:
					break
			#print('epoch:%d'%epoch)
		time.sleep(60)
		with open(summaryfile,'a+') as f:

			header = ['settings','end_epochs','valaupr','valauroc','testaupr','testauroc','tn','fp','fn','tp']
			f.writelines(','.join(header)+'\n')
			#settins = '-'.join([param + ':' + str(args[param]) for param in args])
			#settings = logfile
			contents = [
				model_name+'_'+'-'.join(['%d'%batch_size,'%d'%rnn_layers,'%d'%h_dim,'%d'%z_dim,'%.2f'%wd]),
				'%d'%end_epoch,
				'%.6f'%best_val_aupr,
				'%.6f'%val_auroc,
				' %.4f'%test_aupr,
				'%.4f'%test_roc,
				'%.4f'%test_tn,
				'%.4f'%test_fp,
				'%.4f'%test_fn,
				'%.4f'%test_tp
			]

			f.writelines(','.join(contents)+'\n')




