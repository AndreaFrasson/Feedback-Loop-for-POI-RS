from preprocess import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop
import json

# SETTINGS GENERAL RECOMMENDER
MODEL = 'MultiVAE'
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
    if n < 3:
        len_step = 5
        epochs = 20
    else:
        len_step = int(sys.argv[1])
        epochs = int(sys.argv[2])

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
            'learnign_rate': LEARNING_RATE,
        }


    fl = FeedBack_Loop(config_dict)
    fl.loop(epochs, len_step, user_frac=1, hyper_file='MultiVAE.hyper', tuning=True)

        # save output
    with open('output/'+fl.config['model']+'_'+str(m)+'-'+str(epochs)+'.txt','w') as data:  
      json.dump(fl.metrics, data)
