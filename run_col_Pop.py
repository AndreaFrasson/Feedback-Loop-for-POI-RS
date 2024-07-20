from preprocess import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop
import json

# SETTINGS
MODEL = 'Pop'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 20
DEVICE_ID = '0'

# Default parameters



if __name__ == '__main__':
    # total arguments
    if n < 3:
        len_step = 5
        epochs = 20
    if n < 4:
        len_step = int(sys.argv[1])
        epochs = int(sys.argv[2])
    else:
        len_step = int(sys.argv[1])
        epochs = int(sys.argv[2])
        not_rec = sys.argv[2]

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


    fl = FeedBack_Loop(config_dict)
    fl.loop(epochs, len_step, user_frac=0, tuning=False)

        # save output
    with open('output/'+config_dict['model']+'_'+str(len_step)+'-'+str(epochs)+'.txt','w') as data:  
      json.dump(fl.metrics, data)