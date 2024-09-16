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
MODEL = 'uCF'
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
    fl.initialize()


    user_frac = 0.2
    fl.config['learning_rate'] = 0.007445981808674969
    
    fl.model = get_model(fl.config['model'])(fl.config, fl.training_set._dataset).to(fl.config['device'])
    # trainer loading and initialization
    fl.trainer = get_trainer(fl.config['MODEL_TYPE'], fl.config_dict['model'])(fl.config, fl.model)
    # model training
    best_valid_score, best_valid_result = fl.trainer.fit(fl.training_set, fl.validation_set)

    if user_frac < 1:
                
        users = list(fl.training_set._dataset.user_counter.keys())
        user_num = len(users) / len_step

        np.random.seed()
        user_not_active = np.random.choice(users,
                                    int(user_num)) 
        
        rows_not_active = user_not_active - 1 

    else:
        rows_not_active = None
    
    rows_not_active = None

    #recommender choices
    rec_scores, rec_predictions = fl.generate_prediction(fl.training_set._dataset, rows_not_active)

    output = []
    #output.append(list(rec_predictions))


    for j in range(5):
        print(j)

        for i in range(len_step):
            #recommender choices
            rec_scores, rec_predictions = fl.generate_prediction(fl.training_set._dataset, rows_not_active)
            #output.append(list(rec_predictions))
            
            # not recommender choices
            not_rec_predictions = fl.generate_not_rec_predictions()
            
            # choose one item
            chosen_items = fl.choose_items(rec_predictions, rec_scores, not_rec_predictions, rows_not_active, 0.5)

            fl.update_incremental(chosen_items)
        
        rec_scores, rec_predictions = fl.generate_prediction(fl.training_set._dataset, rows_not_active)
        output.append(list(rec_predictions))

        fl.model = get_model(fl.config['model'])(fl.config, fl.training_set._dataset).to(fl.config['device'])
        # trainer loading and initialization
        fl.trainer = get_trainer(fl.config['MODEL_TYPE'], fl.config_dict['model'])(fl.config, fl.model)
        # model training
        best_valid_score, best_valid_result = fl.trainer.fit(fl.training_set, fl.validation_set)


        rec_scores, rec_predictions = fl.generate_prediction(fl.training_set._dataset, rows_not_active)
        output.append(list(rec_predictions))

    # save output

    with open('dataframe/'+MODEL, 'wb') as fp:
        pickle.dump(output, fp)