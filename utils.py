import numpy as np
from recbole.data import data_preparation, create_dataset
from recbole.utils import init_seed
from recbole.data.interaction import Interaction
import torch
import os
from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer
import pandas as pd
import copy
import pickle


def _from_ids_to_int(items):
    return np.array([int(i) for i in items])


def _from_int_tp_str(items):
    return np.array([str(i) for i in items])


# Hyperparameter tuning for the model. Perform a random search for
# tuning the hyperparameter for the model
# @input model (string): name of the model to search
# @input hyper_file (string): name of the file with the values of the hyperparameter
# @return params (dict): best parameters found  
def tuning(model_name, hyper_file, config_dict = {}):


    def objective_function(params_dict=None, config_file_list=None):

        config = Config(config_dict=config_dict, config_file_list=['environment.yaml'])

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
                max_evals=100, params_file=hyper_file, fixed_config_file_list=['environment.yaml'], params_dict=config_dict)

    hp.run()
    params =  config_dict | hp.best_params
    return params



# given a list of users, a matrix of items visited, and the model make the prediction for each user.
# @input users: list, internal user ids
# @input items: torch.Tensor, interactions between users and items. Should be reshaped to 
#               match the shape (len(users), len(set(items)))
# @input model: recbole.model.*, model
# @return only the 10 best items, INTERNAL EMBEDDING, predicted for each user
def prediction(users, items, model):

    if not torch.is_tensor(users):
        torch_users = torch.tensor(copy.deepcopy(users))
    else:
        torch_users = users

    #make prediction for users
    input_inter = Interaction({
        'uid': torch_users,
        'venue_id': items.reshape(len(users), -1),
    })

    with torch.no_grad():
        scores = model.full_sort_predict(input_inter).reshape((len(users), -1))
    
    # get the 10 items with highest scores
    rec_list = np.argsort(scores, axis = 1)[:, -10:]

    return rec_list



# Update the interaction files (for training) in an incremental fashion.
def update_incremental(new_items, training_set, validation_set):

    iid_field = training_set.dataset.iid_field
    uid_field = training_set.dataset.uid_field

    # open the training file
    train_df = pd.read_csv('foursquare/foursquare.part1.inter', sep = ',')
    # open the validation file
    valid_df = pd.read_csv('foursquare/foursquare.part2.inter', sep = ',')

    # new training: old training + old validation
    new_train = pd.concat([train_df, valid_df]).sort_values(by = ['uid:token', 'timestamp:token'])
    # save new file and new interactions in the dataloader
    new_train.to_csv('foursquare/foursquare.part1.inter', sep = ',', index=False)
    new_train.columns = ['uid', 'venue_id', 'timestamp']
    
    # update training set with the internal id values
    for c in [uid_field, iid_field]:
        tokens = [str(i) for i in new_train[c].to_numpy()]
        new_train[c] = training_set.dataset.token2id(c, tokens)
    
    training_set._dataset.inter_feat = Interaction(new_train.copy())

    # new validation: new predicted values
    tokens = valid_df['uid:token'].to_numpy()
    timestamps = max(valid_df['timestamp:token'].to_numpy()) + 1

    # make rows (uid, item, timestamp) for the new validation file
    new_val = pd.DataFrame(zip(tokens, new_items, [timestamps] * len(tokens)),
                columns=['uid:token', 'venue_id:token', 'timestamp:token'])
    new_val = new_val.sort_values(['uid:token', 'timestamp:token'])
    new_val.to_csv('foursquare/foursquare.part2.inter', sep = ',', index=False)

    new_val.columns = ['uid', 'venue_id', 'timestamp']

    for c in [uid_field, iid_field]:
        tokens = [str(i) for i in new_val[c].to_numpy()]
        new_val[c] = validation_set.dataset.token2id(c, tokens)

    validation_set._dataset.inter_feat = Interaction(new_train.copy())

    

def _from_ids_to_category(items, id_cat_dict):
    return np.array([id_cat_dict[int(i)] for i in items])


def _get_category_distribution_by_user(interactions, id_cat_dict):
    category_interactions = _from_ids_to_category(interactions, id_cat_dict)

    keys = list(id_cat_dict.keys())
    probability_dict = dict(zip(keys, [0.00001] * len(keys)))

    for k in category_interactions:
        pr = len(np.where(category_interactions == k)[0])/len(category_interactions) + 0.00001
        probability_dict[k] = pr
    
    return list(probability_dict.values())


# Function to choose one item from the recommandation list.
# First, from the visits it extract the probability that a user visits a category based on the history.
# Then, sample one item in the recommandation list, using the probability associated with the category.
# @input rec_list
# @input category_dict
# @input visits
# #output item (int): id of the selected item
def choose_item(external_item_list, dataset, mode = 'random'):

    if mode == 'random' or mode == 'r':
        return np.apply_along_axis(np.random.choice, 1, external_item_list)
    
    if mode == 'category' or mode == 'c':
        with open('id_category.pkl', 'rb') as file:
            loaded_dict = pickle.load(file)

        iid_field = dataset.iid_field
        uid_field = dataset.uid_field

        users = np.unique(dataset.id2token(dataset.uid_field, dataset.inter_feat[uid_field]))


        interactions = dataset.id2token(dataset.iid_field, dataset.inter_feat[iid_field]).reshape(len(users), -1)
        interactions = np.apply_along_axis(_from_ids_to_int, 1, interactions)

        probability = np.apply_along_axis(_get_category_distribution_by_user, 1, interactions, id_cat_dict = loaded_dict)

        selected_items = []
        for items, prob in zip(external_item_list, probability):
            category_recommended = _from_ids_to_category(items, id_cat_dict = loaded_dict)
            prob_distr = [prob[i] for i in category_recommended]
            prob_distr_norm = np.array(prob_distr) / sum(prob_distr)

            selected_items.append(np.random.choice(items, p = prob_distr_norm))

        return selected_items

    else:
        raise NotImplementedError


