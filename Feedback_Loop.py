import utils
from preprocess import preprocess
import metrics
import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import torch
import copy
from recbole.data import Interaction
import pickle


class FeedBack_Loop():

    def __init__(self, config, steps):
        self.config = config
        self.steps = steps 
        self.__initialize()


    def __initialize(self):
        self.dataset = create_dataset(self.config)

        self.iid_field = self.dataset.iid_field
        self.uid_field = self.dataset.uid_field
        self.time_field = self.dataset.time_field

        self.training_set, self.validation_set, self.test_set = data_preparation(self.config, self.dataset)

        # get model
        self.model = get_model(self.config['model'])(self.config, self.training_set.dataset).to(self.config['device'])
        # trainer loading and initialization
        self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
        return 
    

    def loop(self, MaxIt):

        for c in tqdm(range(1,MaxIt+1)):
            if c % self.steps == 0:
                self.__initialize()
                # model training
                # get model
                self.model = get_model(self.config['model'])(self.config, self.training_set.dataset).to(self.config['device'])
                # trainer loading and initialization
                self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
                best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
                print(best_valid_score)
                results = self.trainer.evaluate(self.test_set)

            predictions = self.generate_prediction(self.training_set._dataset)
            # choose one item
            chosen_items = utils.choose_item(predictions, self.training_set._dataset, 'c')

            self.update_incremental(chosen_items)
            



    
    def fit(self):
        best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
        return best_valid_score, best_valid_result
    

    def evaluate(self):
        results = self.trainer.evaluate(self.test_set)
        return results



    def generate_prediction(self, dataset):
        users = list(dataset.user_counter.keys())

        # generate synthetic data for the training
        users = torch.tensor(copy.deepcopy(users))
        items = dataset.inter_feat[dataset.iid_field].reshape(len(users), -1)
        rec_list = self.__prediction(users, items, self.model)

        # translate the location id back to the original embedding
        external_item_list = dataset.id2token(dataset.iid_field, rec_list.cpu())     
        
        return np.apply_along_axis(utils._from_ids_to_int, 1, external_item_list)
    

    # given a list of users, a matrix of items visited, and the model make the prediction for each user.
    # @input users: list, internal user ids
    # @input items: torch.Tensor, interactions between users and items. Should be reshaped to 
    #               match the shape (len(users), len(set(items)))
    # @input model: recbole.model.*, model
    # @return only the 10 best items, INTERNAL EMBEDDING, predicted for each user
    def __prediction(self, users, items, model):


        #make prediction for users
        input_inter = Interaction({
            'uid': users,
            'venue_id': items.reshape(len(users), -1),
        })

        with torch.no_grad():
            scores = model.full_sort_predict(input_inter).cpu().reshape((len(users), -1))
        
        # get the 10 items with highest scores
        rec_list = np.argsort(scores, axis = 1)[:, -10:]

        return rec_list
    


    # Update the interaction files (for training) in an incremental fashion.
    def update_incremental(self, new_items):

        training_df = pd.DataFrame(self.training_set._dataset.inter_feat.numpy())
        validation_df = pd.DataFrame(self.validation_set._dataset.inter_feat.numpy())

        new_train = pd.concat([training_df, validation_df]).sort_values(by = [self.uid_field, self.time_field])
        self.training_set._dataset.inter_feat = Interaction(new_train.copy(deep = True))

        new_timestamp = self.validation_set._dataset.inter_feat['timestamp'].cpu().numpy() + 1
        valid_user = self.validation_set._dataset.inter_feat['uid'].cpu().numpy()

        new_valid = pd.DataFrame(list(zip(valid_user, new_items, new_timestamp)), 
                     columns=[self.uid_field, self.iid_field, self.time_field])
        
        self.validation_set._dataset.inter_feat = Interaction(new_valid.copy(deep=True))