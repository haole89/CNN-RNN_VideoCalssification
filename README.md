# CNN-RNN_VideoCalssification

## Prerequisites
pip install -r requirements.txt

## Step 1: Data collection

!wget -q https://git.io/JGc31 -O ucf101_top5.tar.gz

!tar xf ucf101_top5.tar.gz

## Step 2: Features extraction using InceptionV3 pre-trainned model
The feature dimension is 2048 and the maximun sequence frames can be set by the MAX_SEQ_LENGH parameter
This process will take ~20 minutes to execute depending on the machine it's being executed.
The results will save in numpy format for speedup training RNN model

$ python features_extractor.py


## Step 3: Train RNN model 
## The model save in hd5 format in the folder: data/checkpoints/lstm.model
## The labels save in binary formate in the folder: data/checkpoints/tag.pickle

$ python train.py

## Step 4: Testing 

$ python clasify.py



