import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from model_utils import init_weights,kld_guass_exact,sigma_nll,hard_sigmoid
from torch import linalg as LA


class VRNN2(nn.Module):
	def __init__(self, input_dim, rnn_h_dim, rnn_n_layers, z_dim, rnn_dropout=0.0,  rnn_bias = False, model_type=None):
		super(VRNN2, self).__init__()
		self.input_dim=input_dim
		self.rnn_h_dim=rnn_h_dim
		self.rnn_n_layers=rnn_n_layers
		self.z_dim=z_dim
		self.model_type=model_type
		self.rnn_dropout=rnn_dropout
		self.rnn_bias = rnn_bias

		self.rnn_input_dim = self.input_dim
		self.rnn = nn.GRU(self.rnn_input_dim+self.z_dim, self.rnn_h_dim, self.rnn_n_layers, self.rnn_bias, batch_first = True)

		#clf
		self.clf = nn.Sequential(nn.Linear(self.z_dim, 1))

		#prior based on h(t-1)
		self.prior_mean = nn.Sequential(
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim, self.rnn_h_dim*2),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*2, self.z_dim)
			)
		self.prior_logvar = nn.Sequential(
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim, self.rnn_h_dim*2),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*2, self.z_dim)
			)

		#encoder based on x(t) and h(t-1)
		self.enc_mean = nn.Sequential(
			nn.ReLU(),
			nn.Linear(self.input_dim+self.rnn_h_dim, self.rnn_h_dim*3),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*3, self.z_dim)
			)
		self.enc_logvar = nn.Sequential(
			nn.ReLU(),
			nn.Linear(self.input_dim+self.rnn_h_dim, self.rnn_h_dim*3),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*3, self.z_dim)
			)

		#decoder based on z(t) and h(t-1)
		self.dec_mean = nn.Sequential(
			nn.Linear(self.rnn_h_dim+self.z_dim, self.rnn_h_dim*2),
			#nn.BatchNorm1d(self.rnn_h_dim*2),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*2, self.rnn_h_dim*4),
			nn.ReLU(),
			nn.Linear(self.rnn_h_dim*4, self.input_dim)
			)

		#init
		self.clf.apply(init_weights)
		self.prior_mean.apply(init_weights)
		self.prior_logvar.apply(init_weights)
		self.enc_mean.apply(init_weights)
		self.enc_logvar.apply(init_weights)
		self.dec_mean.apply(init_weights)

	def get_weights(self):
		w={}
		clf_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.clf) if type(layer)==nn.Linear]
		w['clf_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in clf_weights]))		

		rnn_weights = [LA.matrix_norm(param) for name, param in self.named_parameters() if name.startswith('weight')]
		w['rnn_weights'] = sum(rnn_weights)

		prior_mean_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.prior_mean) if type(layer)==nn.Linear]
		w['prior_mean_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in prior_mean_weights]))
		prior_logvar_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.prior_logvar) if type(layer)==nn.Linear]
		w['prior_logvar_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in prior_logvar_weights]))

		enc_mean_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.enc_mean) if type(layer)==nn.Linear]
		w['enc_mean_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in enc_mean_weights]))
		enc_logvar_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.enc_logvar) if type(layer)==nn.Linear]
		w['enc_logvar_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in enc_logvar_weights]))

		dec_mean_weights = [[s.view(-1) for s in layer.parameters()] for i,layer in enumerate(self.dec_mean) if type(layer)==nn.Linear]
		w['dec_mean_weights'] = torch.sum(torch.Tensor([torch.norm(w[0]) for w in dec_mean_weights]))

		return w


	def weight_sum(self):
		w=self.get_weights()
		return sum([w[k] for k in w.keys()])

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu+eps.mul(std)

	#sample size n
	def sample_z(self, n, device):
		sample = torch.randn(n, self.rnn_h_dim+self.z_dim).to(device)
		return self.decode(sample)	


	#input shape: [batch_size, seq_len, input_dim]
	#mask: [batch_size, max_len]
	def forward(self, x, mask=None,is_train=False):
		# vrnn init
		
		batch_size,max_len = x.shape[0], x.shape[1]
		mask=torch.ones(batch_size,max_len).to(x.device)
		kld_loss,nll_loss = 0,0
		x_recon,y_pred=[],[]
		zt,logvars,mus=[],[],[]
		h = Variable(torch.zeros(self.rnn_n_layers,batch_size , self.rnn_h_dim)).to(x.device)

		for t in range(max_len):
			#prior
			prior_mean_t=self.prior_mean(h[-1])
			prior_logvar_t=self.prior_logvar(h[-1])

			x_t = x[:,t,:]
			#encoder
			enc_x = torch.cat([x_t, h[-1]], 1)
			enc_mean_t = self.enc_mean(enc_x)
			enc_logvar_t = self.enc_logvar(enc_x)
			logvars.append(enc_logvar_t)
			#z, reparameterization
			z_t=self.reparameterize(enc_mean_t, enc_logvar_t)
			zt.append(z_t)
			mus.append(enc_mean_t)
			if is_train:
				y_pred.append(self.clf(z_t))
			else:
				y_pred.append(self.clf(enc_mean_t))
			

			dec_x = torch.cat([z_t, h[-1]], 1)
			dec_mean_t = self.dec_mean(dec_x)
			x_recon.append(dec_mean_t)

			# recurrence
			_, h = self.rnn(torch.cat([x_t, z_t], 1).unsqueeze(1), h)

			mask_t=mask[:,t]
			kld_t=mask_t*kld_guass_exact(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
			nll_t=mask_t*sigma_nll(dec_mean_t, x_t)

			kld_loss+=(kld_t.sum())
			nll_loss+=(nll_t.sum())
		
		res={}
		res['kld_loss']=kld_loss
		res['nll_loss']=nll_loss
		res['y_score']=torch.stack(y_pred).transpose(0,1)
		res['x_recon']=torch.stack(x_recon).transpose(0,1)
		res['logvar']=torch.stack(logvars).transpose(0,1)
		res['z']=torch.stack(zt).transpose(0,1)
		res['mus']=torch.stack(mus).transpose(0,1)		
		return res
