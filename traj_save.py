from preprocess import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop
import json
import pandas as pd
import numpy as np


# SETTINGS GENERAL RECOMMENDER
MODEL = 'ItemKNN'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 20
DEVICE_ID = '0'

# Default parameters
LEARNING_RATE = 0.005



if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 5:
        len_step = 5
        epochs = 20
    else:
        len_step = int(sys.argv[1])
        epochs = int(sys.argv[2])
        not_rec = sys.argv[3]
        k = float(sys.argv[4])

    # make the atomic files form the data
    seed = 1234 # to get always the same users in train/test
    preprocess(seed)


    config_dict = {
            'model': MODEL,
            'data_path': DATA_PATH,
            'top_k': TOP_K,
            'dataset': DATASET,
            'epochs': EPOCHS,
            'use_gpu': len(DEVICE_ID) > 0,
            'gpu_id': DEVICE_ID,
            'learnign_rate': LEARNING_RATE,
        }


    results = {}
    
    fl = FeedBack_Loop(config_dict, not_rec)
    fl.loop(epochs, len_step, k = k, user_frac=0, tuning=False, hyper_file = 'MultiVAE.hyper')

    df = pd.DataFrame(fl.training_set._dataset.inter_feat.numpy())

    # save output
    df.to_csv('dataframe/uCF_dataframe.csv', index = False)