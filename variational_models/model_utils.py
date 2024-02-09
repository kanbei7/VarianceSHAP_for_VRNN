import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

logsqrt2pi=0.5 * np.log(2 * np.pi)


def init_weights(layer):
	if type(layer) == nn.Linear or type(layer) == nn.Conv1d:
		init.xavier_normal_(layer.weight)

def softclip(tensor, min_v):
	result_tensor = min_v + F.softplus(tensor - min_v)
	return result_tensor

def hard_sigmoid(logits):
	y_soft = torch.sigmoid(logits)
	index = (y_soft>=0.5).long()
	y_hard = torch.ones_like(logits, memory_format = torch.legacy_contiguous_format)*index
	return y_hard - y_soft.detach() + y_soft

def nll_gauss(mu, log_sigma, x):
	return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + logsqrt2pi

def kld_guass_exact(mu1, logvar1, mu2, logvar2, ):
	kld = logvar2-logvar1+(logvar1.exp()+(mu1-mu2)**2)/logvar2.exp()-1
	return 0.5*kld.sum(-1)

def kld_standardnormal_exact(logvar, mu):
	kld = mu.pow(2)+logvar.exp()-logvar-1
	return 0.5*kld.sum(-1)

#keep batch dim
def sigma_nll(x_hat, x):
	log_sigma = ((x - x_hat) ** 2).mean(dim=1,keepdim=True).sqrt().log()
	log_sigma = softclip(log_sigma, -6)
	return nll_gauss(x_hat, log_sigma, x).sum(-1)



#functions which work with logvar, instead of logsigma
'''
def nll_gauss(mu, logvar, x, reduction = True):
	nll = 0.5*(((x-mu)**2)/logvar.exp() + logvar)
	nll = nll.sum(-1)
	if reduction:
		nll = torch.mean(nll)
	return nll

#keep batch dim
def kld_guass_exact(logvar1, mu1, logvar2, mu2, reduction = True):
	kld = logvar2-logvar1+(logvar1.exp()+(mu1-mu2)**2)/logvar2.exp()-1
	kld = kld.sum(-1)
	if reduction:
		kld = torch.mean(kld)
	return 0.5*kld

#keep batch dim
def kld_guass(z, mu, std, reduction = True):
	p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
	q = torch.distributions.Normal(mu, std)

	log_qzx = q.log_prob(z)
	log_pz = p.log_prob(z)

	kld = (log_qzx - log_pz)
	kld = kld.sum(-1)
	if reduction:
		kld = torch.mean(kld)
	return kld

#keep batch dim


#keep batch dim
def kld_standardnormal(logvar, mu, reduction = True):
	kld = mu**2+logvar.exp()-logvar-1
	kld = kld.sum(-1)
	if reduction:
		kld = torch.mean(kld)
	return kld

'''




