import utils as u
from preprocess import preprocess
import metrics

import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer

import matplotlib.pyplot as plt

def default_params():
    params ={'model': 'BPR',
        'data_path': '/Users/andreafrasson/Desktop/tesi/Feedback-Loop-for-POI-RS',
        'topk': 10,
        'use_gpu': True,
        'gpu_id': 0,
        'dataset': 'foursquare',
        'embedding_size': 64,
        'learning_rate': 0.0014441770317725243,
        'mlp_hidden_size': [128, 128],
        'train_batch_size': 2048,
        'eval_args': {
            'group_by': 'user',
            'order': 'TO', # temporal order
            'split': {'LS': 'valid_and_test'}, # leave one out
            'mode': 'full'}
    }

    return params

if __name__ == '__main__':

    # make the atomic files form the data
    seed = 1234 # to get always the same users in train/test
    preprocess(seed)

    # train users
    train_users = pd.read_csv('foursquare/foursquare.part1.inter', sep = ',')['uid:token'].to_list()
    train_users = list(set(train_users))

    #hyperparameter tuning
    default = True
    if default:
        params = default_params()
    else:
        params = u.tuning('BPR', 'bpr.hyper', {'dataset': 'foursquare'})

    # create the configurator
    config = Config(config_file_list=['foursquare_general.yaml'], config_dict = params)

    # environment settings dor the loop
    m = 3
    MaxIt = 20
    c = 0

    # output metrics
    hit = []
    prec = []
    card = []
    mean_entropy_train = []

    # Main Loop
    while c < MaxIt:
        print('--------- iteration number: ', c)

        if c % m == 0:  
            # create the dataset
            dataset = create_dataset(config)
            # split
            training_set, validation_set, test_set = data_preparation(config, dataset)

            # get model
            model = get_model(config['model'])(config, training_set.dataset).to(config['device'])

            # trainer loading and initialization
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
            # model training
            best_valid_score, best_valid_result = trainer.fit(training_set, validation_set)
            results = trainer.evaluate(test_set)
            hit.append(results['hit@10'])
            prec.append(results['precision@10'])

            print('Test error: ', results)

        # generate synthetic data for the training
        uid_series = training_set._dataset.token2id(training_set._dataset.uid_field, [str(s) for s in train_users])
        items = training_set._dataset.inter_feat[training_set._dataset.iid_field].reshape(len(train_users), -1)
        rec_list = u.prediction(uid_series, items, model)

        # translate the location id back to the original embedding
        external_item_list = training_set.dataset.id2token(training_set.dataset.iid_field, rec_list.cpu())
        card.append(len(np.unique(external_item_list.flatten())))

        # choose one item
        chosen_items = u.choose_item(rec_list, training_set._dataset, 'c')

        # update the training/validation file
        u.update_incremental(chosen_items, training_set, validation_set)

        # compute the entropy for the new training set
        entropy_train = metrics.uncorrelated_entropy(pd.read_csv('foursquare/foursquare.part1.inter', sep = ','))
        # mean value for the users
        mean_entropy_train.append(np.mean(entropy_train['entropy'].to_numpy()))

        c+=1
        print('')