from preprocess_sequential import preprocess
import os
import sys
from Feedback_Loop import FeedBack_Loop

# SETTINGS
MODEL = 'Caser'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 30
DEVICE_ID = '0'

# Default parameters
LEARNING_RATE = 0.0014441770317725243
REG_WEIGHTS = 1e-4
DROPOUT_PROB = 0.1


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
            'learnign_rate': LEARNING_RATE,
            'reg_weights': REG_WEIGHTS,
            'drpout_prob':DROPOUT_PROB,
            'train_neg_sample_args': None,
            'LIST_SUFFIX': '_list',
            'MAX_ITEM_LIST_LENGTH': 50,
            'load_col': {
                'inter': ['uid', 'item_id', 'timestamp'],
                'item': ['venue_id','lat', 'lon','venue_category_name']
            },
    }


    fl = FeedBack_Loop(config_dict, m)
    fl.loop(MaxIt, 'r', True, 'Caser.hyper')

    # save output
    with open('output/'+fl.config['model']+'_'+str(m)+'-'+str(MaxIt)+'.txt','w') as data:  
      data.write(str(fl.metrics))