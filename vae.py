import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from data import *


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, data_dim, constrain_mean = False, *args):
        super(VAE,self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.log_var = nn.Sequential(
            nn.Linear(hidden_dim,data_dim),
        )
        if constrain_mean:
            self.mu = nn.Sequential(
                nn.Linear(hidden_dim,data_dim),
                nn.Sigmoid(),
            )
        else:
            self.mu = nn.Sequential(
                
                    nn.Linear(hidden_dim,data_dim),
                )
            
         
    def get_mean_and_log_var(self,x):
        #print(f"Input shape: {x.shape}")
        h = self.h(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu,log_var
    def forward(self, x, epsilon):
        mu,log_var = self.get_mean_and_log_var(x) 
        sigma = torch.sqrt(torch.exp(log_var))
        return mu + sigma * epsilon
    def compute_log_density(self, y, x):
        '''
        Compute log p(y|x)
        '''
        mu, log_var = self.get_mean_and_log_var(x)
        log_density = -.5 * (torch.log(2 * torch.tensor(np.pi)) + log_var + (((y-mu)**2)/(torch.exp(log_var) + 1e-10))).sum(dim=1)
        return log_density
    def compute_KL(self, x):
        '''
        Assume that p(x) is a normal gaussian distribution; N(0, 1)
        '''
        mu, log_var = self.get_mean_and_log_var(x)
        return -.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)



def AVEB(data, encoder,decoder, encoder_optimizer, decoder_optimizer, epochs, device='cpu',M=1001,L=1, latent_dim = 2):
    losses = []
    for _ in tqdm(range(epochs)):
        x = get_minibatch(data,M, device=device)
        epsilon = torch.normal(torch.zeros(M * L, latent_dim), torch.ones(latent_dim)).to(device)

        # Compute the loss
        z = encoder(x,epsilon)
        log_likelihoods = decoder.compute_log_density(x, z)
        kl_divergence = encoder.compute_KL(x)
        loss = (kl_divergence - log_likelihoods.view(-1, L).mean(dim=1)).mean()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        losses.append(loss.item())
    return losses
