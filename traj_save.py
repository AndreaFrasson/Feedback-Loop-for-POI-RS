from preprocess import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop
import json
import pandas as pd
import numpy as np
import pickle
from recbole.quick_start.quick_start import get_model, get_trainer


# SETTINGS GENERAL RECOMMENDER
MODEL = 'MultiVAE'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 20
DEVICE_ID = '0'

# Default parameters
LEARNING_RATE = 0.007445981808674969



if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 5:
        len_step = 5
        epochs = 2
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
    
    fl = FeedBack_Loop(config_dict, 'cp')
    
    rows_not_active = None

    output = []
    #output.append(list(rec_predictions))

    fl.loop(10, 20, k = 0.8, user_frac=1.5, tuning=False)

    # save output

    with open('dataframe/'+MODEL+'_'+str(0.8), 'wb') as fp:
        pickle.dump(fl.sugg, fp)
    
    pd.DataFrame(fl.training_set._dataset.inter_feat.numpy()).to_csv('dataframe/'+MODEL+'_'+str(0.8)+'.csv')