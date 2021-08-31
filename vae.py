import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from data import *

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=2):
        """
            @param input_dim: input dimension of the data
            @param hidden_dim: hidden dimension of the MLPs
            @param latent_dim: output dimension of MLPs
            @param is decoder: is this module was initialized is decoder ?
            @------------------@
            @return: None
        """
        super(VAE,self).__init__()
        self.en = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim,latent_dim)
        
        self.de = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

        self.final_layer=nn.Sequential(
            nn.Linear(hidden_dim,input_dim),
            nn.Sigmoid()
        )

        
          
    def encode(self,x):
        #x = torch.flatten(x)
        res = self.en(x)
        mu = self.mu(res)
        log_var = self.var(res)
        
        return mu,log_var
            
    def decode(self,x):
        res = self.de(x)
        res = self.final_layer(res)
        return res
    def reparameterize(self,mu,log_var):
        epsilon = torch.normal(mu,torch.exp(0.2 *log_var))
        return mu + log_var * epsilon
    def forward(self,x):
        mu, log_var = self.encode(x)
        norm = self.reparameterize(mu,log_var)
        res = self.decode(norm)
        return (res, x, mu,log_var)
    def loss_fc(self,x,*args):
        (res, x, mu, log_var) = self.forward(x)
        recon_loss = F.mse_loss(x,res)
        KL_divergence = torch.mean(-0.5 * torch.sum((1 + 2 * log_var - mu**2 + torch.exp(log_var)),dim=1), dim=0)
        loss = recon_loss + KL_divergence 
        return dict({'loss': loss, 'recon_loss': recon_loss, 'kl_loss': KL_divergence})
    def generate(self,x):
        return self.forward(x)[0]
