# Auto-Encoding Variational Bayes

_Original Paper: [link](https://arxiv.org/abs/1312.6114)_


## Installation
* Recommend using an virtual environment to run
```bash
pip install -r requirements.txt
```

## Run

### Data set
Go to [Kaggle MNIST Dataset](https://www.kaggle.com/avnishnish/mnist-original) and download
Extract data file to get `mnist.mat`data file.

###### For Linux Shell

```shell
unzip archive.zip 
```


### Start to train the encoder and decoder
```shell
python train.py [OPTIONS] [VALUE]
usage: train.py [-h] -d DATA [-hd HIDDEN] [-ld LATENT] [-lr LEARNING] [-e EPOCHS] [-b BATCH_SIZE]

optional arguments:
  -h, --help            Show this help message and exit
  -d DATA, --data DATA  path/to/train/data
  -hd HIDDEN, --hidden HIDDEN
                        Number of hidden unit
  -ld LATENT, --latent LATENT
                        Number of latent unit
  -lr LEARNING, --learning LEARNING
                        Learning rate
  -e EPOCHS, --epochs EPOCHS
                        Epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
```


