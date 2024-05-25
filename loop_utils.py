import numpy as np
from recbole.data import data_preparation, create_dataset
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.data.interaction import Interaction
import torch
import os
from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer
import pandas as pd



# hyperparameter tuning for the model. Perform a random search for
# tuning the hyperparameter for the model
# @input model (string): name of the model to search
# @input hyper_file (string): name of the file with the values of the hyperparameter
# @output params (dict): best parameters found  
def tuning(model_name, hyper_file, params_dict = {}):

    config_dict = {
            'model': model_name,
            'data_path': os.getcwd(),
            'topk': 10,
            'use_gpu': True,
            'gpu_id': '0',
            'seed': 1234
    }
    config_dict.update(params_dict)

    def objective_function(params_dict=None, config_file_list=None):
        k = 10

        config = Config(config_dict=config_dict, config_file_list=['foursquare_general.yaml'])

        if config['seed'] is not None and config['reproducibility'] is not None:
            init_seed(config['seed'], config['reproducibility'])

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model_name = config['model']
        model = get_model(model_name)(config, train_data._dataset)
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        test_result = trainer.evaluate(test_data)

        return {
            'model': model_name,
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }

    hp = HyperTuning(objective_function=objective_function, algo='random', early_stop=10,
                    max_evals=100, params_file=hyper_file, fixed_config_file_list=['foursquare_general.yaml'], params_dict=config_dict)

    hp.run()
    params = hp.best_params | params_dict
    return params


# train the model with the specified parameters
# @input model (string): name of the model to train
# @input params (dict): parameters
# @output model (): trained model
# @output test_results (OrderedDict): Hit@10 and Precision@10 obtained in the training
def training(model, params, train_data, valid_data):
    k = 10

    config_dict = {
            'model': model,
            'data_path': os.getcwd(),
            'topk': k,
            'use_gpu': True
    }
    config_dict.update(params)
    # configurations initialization

    config = Config(model=model, dataset='foursquare', config_file_list=['foursquare_general.yaml'], config_dict = config_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering
    #dataset = create_dataset(config)

    # dataset splitting
    #train_data, valid_data, test_data = data_preparation(config, dataset)

    # train the model
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    return trainer


# for the interactions between users and items. Generate a matrix where the rows
# are the items selected for each user.
# @input inter (Pandas DataFrame): interaction dataframe
# @output visits (NumPy Matrix): matrix with all the visited places for each user.
def get_history(interactions):
    visits = []
    set_uid = set(interactions['uid:token'])
    for u in set_uid:
        visits.append(interactions[interactions['uid:token'] == u]['venue_id:token'].values.tolist())
    
    return np.array(visits)


# function to choose one item from the recommandation list.
# First, from the visits it extract the probability that a user visits a category based on the history.
# Then, sample one item in the recommandation list, using the probability associated with the category.
# @input rec_list
# @input category_dict
# @input visits
# #output item (int): id of the selected item
def choose_item(rec_list, category_dict, visits):
    p_v = []
    num_category = len(set(category_dict.values()))

    # transform POI visits in category visits
    visited_category = []
    for i in visits:
        visited_category.append(category_dict[i])

    # get distribution over recommended items
    for i in rec_list:
        category = category_dict[int(i)]
        p = (len(np.where(np.array(visited_category) == int(category))[0]) + 1) / (len(visits) + len(category_dict) + num_category +1)
        #extract the probability associated with each category for that users
        p_v.append(p)

    # extract poi based on the popularity
    p = np.array(p_v)/sum(p_v)
    return int(np.random.choice(rec_list, p = p))



def get_prediction(model, interactions, items):
    k = 10

    visits = get_history(interactions)

    unique_users = list(set(interactions['uid:token']))

    #make prediction for users
    input_inter = Interaction({
        'uid': torch.tensor(unique_users),
        'venue_id': torch.tensor(visits)
    })

    with torch.no_grad():
        scores = model.full_sort_predict(input_inter).reshape((len(unique_users), -1))

    #length |items| + 1 because of the padding

    # get the 10 items with highest scores
    rec_list = np.argsort(scores, axis = 1)[:, -k:]

    # select one item in the list
    # id-category for POI dictionary
    id_cat = dict(zip(items['venue_id:token'], items['venue_category_name:token']))

    rec_item = []
    for i in range(len(rec_list)):
        rec_item.append(choose_item(rec_list[i], id_cat, visits[i]))

    return rec_item



def update_interactions(users, predictions, interactions):
    current_time = max(interactions['timestamp:token'])+1
    
    new_locations = pd.DataFrame({'uid:token': users, 'venue_id:token':predictions, 'timestamp:token':[current_time]*len(predictions)}, columns=['uid:token', 'venue_id:token', 'timestamp:token'])

    interactions = pd.concat([interactions, new_locations], axis = 0).reset_index(drop = True)
    interactions.sort_values(by=['uid:token', 'timestamp:token'], inplace=True)

    interactions.to_csv('foursquare/foursquare.inter', index=False, sep = '\t')
