import torch
from data import *
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.optim import Adagrad
from torch.optim import Adam
from tqdm import tqdm
from vae import *
import argparse
device = "cuda:0"



# Parse agrs
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data',required=True,help='path/to/train/data')
parser.add_argument('-hd','--hidden',type=int,help='number of hidden unit')
parser.add_argument('-ld','--latent',type=int, help="number of latent unit")
parser.add_argument('-lr','--learning',type=int,help="learning rate")
parser.add_argument('-e', '--epochs', type=int,help='epochs')
parser.add_argument('-b', '--batch_size', type=int,help='Batch size')



args = vars(parser.parse_args())



HIDDEN = (args["hidden"] if args["hidden"] else 32)
LATENT = (args["latent"] if args["latent"] else 2)
LR = (args["learning"] if args["learning"] else 1e-3)
BATCH_SIZE = (args["batch_size"] if args["batch_size"] else 32)
EPOCHS = (args["epochs"] if args["epochs"] else 1)

x = torch.Tensor(get_data(args["data"]))

def AEVB(data, model, optimizer, input_dim, output_dim,epochs, batch_size):
    full_loss, kld, recon = [],[],[]
    for epc in range(epochs):
        fl,kl,rec = 0,0,0
        for _ in tqdm(range(x.shape[0] // batch_size)):
            batch = get_minibatch(x,batch_size, device)
            optimizer.zero_grad()
 
            losses = model.loss_fc(batch)
        
            fl += losses["loss"]
            kl += losses["kl_loss"]
            rec += losses["recon_loss"] 
        
            losses["loss"].backward()

            optimizer.step()
        full_loss.append(fl); kld.append(kl); recon.append(rec)
        print(f"Epoch {epc}\tFull loss: {full_loss[-1]}")
        
    return model, full_loss, kld, recon 



if __name__=="__main__":
    print(type(x.shape[1]) )
    print(x)
    print(type(HIDDEN))
    model = VAE(x.shape[1],HIDDEN,LATENT).to(device)
    print(x.shape)
    #print(get_minibatch(x,32).shape)
    optimizer = Adam(model.parameters(), lr=LR)
    AEVB(data=x,model=model,optimizer=optimizer,input_dim=x.shape[1],output_dim=x.shape[1], batch_size=BATCH_SIZE, epochs=EPOCHS).to(device)
