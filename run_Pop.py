import utils as u
from preprocess import preprocess
import metrics
import pandas as pd
import numpy as np
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from Feedback_Loop import FeedBack_Loop

# SETTINGS
MODEL = 'Pop'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 30
DEVICE_ID = '0'

# Default parameters


def run(m = 3, MaxIt = 20, tuning = False):

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
            'device_id': DEVICE_ID,
        }


    fl = FeedBack_Loop(config_dict, m)
    fl.loop(MaxIt, 'c', tuning)

    print(fl.metrics['rog_ind'])




if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 3:
        m = 5
        MaxIt = 20
    else:
        m = int(sys.argv[1])
        MaxIt = int(sys.argv[2])

    run(m, MaxIt)