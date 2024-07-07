from preprocess import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop

# SETTINGS CONTEXT BASED
MODEL = 'FNN'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 30
DEVICE_ID = '0'

# Default parameters
LEARNING_RATE = 0.01
DROPOUT_PROB = 0.1
MLP_HIDDEN_SIZE = [128,256,128]


if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 3:
        m = 5
        MaxIt = 20
    else:
        m = int(sys.argv[1])
        MaxIt = int(sys.argv[2])

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

            # label must be present
            'train_neg_sample_args': {
                'uniform': 1
            },

            # no timestamp
            'load_col': {
                'inter': ['uid', 'venue_id'],
                'item': ['venue_id','lat', 'lon', 'venue_category_name']
            },

            'learning_rate': LEARNING_RATE,
            'dropout_prob': DROPOUT_PROB,
            'mlp_hidden_size': MLP_HIDDEN_SIZE
        }


    fl = FeedBack_Loop(config_dict, m)
    fl.loop(MaxIt, 'r', False, 'FNN.hyper')

    # save output
    with open('output/'+fl.config['model']+'_'+str(m)+'-'+str(MaxIt)+'.txt','w') as data:  
      data.write(str(fl.metrics))