import torch
from data import *
import scipy
from scipy.io import loadmat
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.optim import Adagrad
from vae import *
import argparse
device = "cpu"



# Parse agrs
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data',required=True,help='path/to/train/data')
args = vars(parser.parse_args())



img_size = (28,28)
img_data = scipy.io.loadmat(args["data"])["data"]

img_data = img_data.T.reshape((-1,28,28))
trainX = torch.tensor(img_data[:int(0.08 * img_data.shape[0])], dtype=torch.float)/255.
print(trainX[0].shape)

if __name__=="__main__": 
    if torch.cuda.is_available():
        device = 'cuda:0'
    encoder = VAE(data_dim=2, input_dim=img_size[0]*img_size[1], hidden_dim=200).to(device)
    decoder = VAE(data_dim=img_size[0]*img_size[1], input_dim=2, hidden_dim=200, constrain_mean=True).to(device)
    encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=0.001, weight_decay=0.5)
    decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=0.001)

    loss = AVEB(trainX ,encoder, decoder, encoder_optimizer, decoder_optimizer, 10**6, device=device)
    
    plt.figure(figsize=(4, 4))
    plt.plot(100*np.arange(len(loss)), -np.array(loss), c='r', label='AVEB (train)')
    plt.xscale('log')
    plt.xlim([10**5, 10**8])
    plt.ylim(0, 1600)
    plt.title(r'Frey Face, $N_z = 2$', fontsize=15)
    plt.ylabel(r'$\mathcal{L}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig('Imgs/Training_loss.png', bbox_inches="tight")
    plt.show()
    
    grid_size = 10
    xx, yy = norm.ppf(np.meshgrid(np.linspace(0.1, .9, grid_size), np.linspace(0.1, .9, grid_size)))

    fig = plt.figure(figsize=(10, 14), constrained_layout=False)
    grid = fig.add_gridspec(grid_size, grid_size, wspace=0, hspace=0)

    for i in range(grid_size):
        for j in range(grid_size):
            img = decoder.get_mean_and_log_var(torch.tensor([[xx[i, j], yy[i, j]]], device=device, dtype=torch.float))
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(np.clip(img[0].data.cpu().numpy().reshape(img_size[0], img_size[1]), 0, 1), cmap='gray', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('Imgs/Learned_data_manifold.png', bbox_inches="tight") 
    plt.show()
