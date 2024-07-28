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
            'device_id': DEVICE_ID,
        }


    results = {}
    
    for i in range(1):
        fl = FeedBack_Loop(config_dict, not_rec)
        fl.loop(epochs, len_step, k = k, user_frac=0, tuning=False)

        for i in fl.metrics.keys():
            results[i] = results.get(i, []) + [fl.metrics[i]]



    # save output
    with open('output/col_Pop_'+not_rec+'_'+str(len_step)+'-'+str(epochs)+'_'+str(k)+'.txt','w') as data:  
        json.dump(results, data)