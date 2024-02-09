import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad,Variable
import shap

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
'''
e = shap.GradientExplainer(model, [background])
svz = e.shap_values([test_data])
''''