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
python train.py --data "path/to/data/file"
```


