import utils
import metrics
import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer
from recbole.utils import init_seed
from tqdm import tqdm
import torch
import copy
from recbole.data import Interaction
from recbole.trainer import HyperTuning
from ind_Random import ind_Random
from ind_Pop import ind_Pop
from user_CF import uCF
from item_CF import iCF

import warnings
warnings.filterwarnings("ignore")

class FeedBack_Loop():

    # initialiazation of the class Feedback loop
    # save the original configuration and the 'control strategy' (what and individual does if 
    # he/she doesn't follow the recommender)
    def __init__(self, config_dict, not_rec = 'ir'):
        self.config_dict = config_dict
        self.not_rec = not_rec

        # if we build a custom model, it returns an error, intercept and build a generic pop model, but save the chosen
        # model for later
        try: 
            self.config = Config(config_file_list=['environment.yaml'], config_dict = config_dict)

        except:
            my_model = copy.copy(self.config_dict['model'])
            self.config_dict['model'] = 'Pop'
            self.config = Config(config_file_list=['environment.yaml'], config_dict = config_dict)
            self.config_dict['model'] = my_model
            


    # first creation of the dataset and set the id variables. Split in training, validation and test 
    # with the configuration settings
    def initialize(self):
        self.dataset = create_dataset(self.config)

        self.iid_field = self.dataset.iid_field
        self.uid_field = self.dataset.uid_field
        self.time_field = self.dataset.time_field

        self.training_set, self.validation_set, self.test_set = data_preparation(self.config, self.dataset)
        self.adj_mat = pd.crosstab(self.training_set._dataset.inter_feat.numpy()[self.uid_field], 
                                   self.training_set._dataset.inter_feat.numpy()[self.iid_field]).transpose()
    

    # Main Loop 
    # if not specified, the model uses default values, otherwise before going in the loop it performs
    # a random search to tune the hyperparameters (dataset splitted in train-val-test).  
    def loop(self, epochs, len_step, k = 0.3, tuning = False, hyper_file = None, user_frac = 0.2):

        self.epochs = epochs
        self.len_step = len_step
        self.metrics = {}

        if tuning:
            if not isinstance(hyper_file, str):
                raise NotImplementedError
            
            self.tuning(hyper_file)

        self.initialize()

        # since the experiment is build as 
        # for each epoch do:
        #   n = 0
        #   while n < simulation_steps:
        #     ....
        # iterations is a variable to save the total number of iterations to use a single for
        iterations = self.epochs * self.len_step

        for c in tqdm(range(iterations)):

            # every len_step epochs, retrain of the model
            if c % self.len_step == 0:
                # get model
                if self.config_dict['model'] == 'ind_Random':
                    self.model = ind_Random(self.config, self.dataset).to(self.config['device'])
                    results = self.model.evaluate(self.test_set._dataset)

                elif self.config_dict['model'] == 'ind_Pop':
                    self.model = ind_Pop(self.config, self.dataset).to(self.config['device'])
                    results = self.model.evaluate(self.test_set._dataset)
                
                elif self.config_dict['model'] == 'uCF':
                    self.model = uCF(self.config, self.dataset).to(self.config['device'])
                    results = self.model.evaluate(self.test_set._dataset)
                    print(np.sum(self.training_set._dataset.inter_matrix().toarray()))
                    print(np.sum(self.test_set._dataset.inter_matrix().toarray()))

                elif self.config_dict['model'] == 'iCF':
                    self.model = iCF(self.config, self.training_set.dataset).to(self.config['device'])
                    results = self.model.evaluate(self.test_set._dataset)
                
                else:
                    self.model = get_model(self.config['model'])(self.config, self.training_set._dataset).to(self.config['device'])
                    # trainer loading and initialization
                    self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config_dict['model'])(self.config, self.model)
                    # model training
                    best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
                    results = self.trainer.evaluate(self.test_set)

                self.metrics['test_hit'] = self.metrics.get('test_hit', []) + [results['hit@10']]
                self.metrics['test_precision'] = self.metrics.get('test_precision', []) + [results['precision@10']]
                self.metrics['test_rec'] = self.metrics.get('test_rec', []) + [results['recall@10']]
                
            
            
            
            #extract user that will not see the recommendations
            if user_frac < 1:
                
                users = list(self.training_set._dataset.user_counter.keys())
                user_num = len(users) / self.len_step

                np.random.seed()
                user_not_active = np.random.choice(users,
                                            int(user_num)) 
                
                rows_not_active = user_not_active - 1 

            else:
                rows_not_active = None

            #recommender choices
            rec_predictions = self.generate_prediction(self.training_set._dataset, rows_not_active)
            
            # not recommender choices
            not_rec_predictions = self.generate_not_rec_predictions()
            
            # choose one item
            chosen_items = self.choose_items(rec_predictions, not_rec_predictions, rows_not_active, k)
                
            if c % self.len_step == 0:
                self.compute_metrics(rec_predictions) # compute metrics before the effect of the new model

            self.update_incremental(chosen_items)


            

    # perform a fit step
    def fit(self):
        best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
        return best_valid_score, best_valid_result
    

    # evaluate the model using the test set
    def evaluate(self):
        results = self.trainer.evaluate(self.test_set)
        return results
    


    def generate_prediction(self, dataset, row_not_active = None):

        torch.cuda.empty_cache()
        scores = self.__prediction()

        if row_not_active is not None:
            scores[row_not_active] = -1

        return scores
    

    # given a list of users, a matrix of items visited, and the model make the prediction for each user.
    # @return only the 10 best items, INTERNAL EMBEDDING, predicted for each user
    def __prediction(self):

        users = torch.unique(self.training_set._dataset.inter_feat[self.uid_field])


        # get the score of every item
        input_inter = Interaction({
            'uid': torch.tensor(users)
        })
        #input_inter = self.dataset.join(input_inter)  # join user feature
    

        with torch.no_grad():
            try:  # if model have full sort predict
                input_inter.to(self.model.device)
                scores = self.model.full_sort_predict(input_inter).cpu().reshape((len(users), -1))

            except NotImplementedError:  # if model do not have full sort predict --> context-aware
                # get feature in the interactions
                raise NotImplementedError
            
            if self.config_dict['model'] != 'uCF' and self.config_dict['model'] !='iCF':
                scores = scores.view(-1, self.dataset.item_num)
                # get the 10 items with highest scores
                #rec_list = np.argsort(scores, axis = 1)[:, -10:]
                topk_score, topk_iid_list  = torch.topk(scores, 10)

                return topk_iid_list.numpy().flatten().reshape(-1,10)
        
        return scores
    
    

    def choose_items(self, recommender_pred, not_recommender_pred, rows_not_active, k = 0.3):
        # percentuale utenti che non seguono il recommender

        users = set(self.training_set._dataset.user_counter.keys())

        if rows_not_active is not None:
            active_users = users - set(rows_not_active)
        else:
            active_users = users

        not_active_users = np.random.choice(list(active_users), int(len(active_users) * k))
        not_active_rows = not_active_users - 1

        choices = np.apply_along_axis(np.random.choice, 1, recommender_pred, size = 1).flatten()
        #random_item_history = [int(np.random.choice(x, 1,)) for x in not_recommender_pred]

        np.array(choices)[not_active_rows] = not_recommender_pred[not_active_rows]

        return choices



    # Update the interaction files (for training) in an incremental fashion.
    def update_incremental(self, new_items):
        new_valid_dict = {}
        new_training_dict = {}

        for u,g in pd.DataFrame(self.validation_set._dataset.inter_feat.numpy()).groupby('uid'):

            try:
                timestamp = int(g[self.time_field].values)
            except:
                timestamp = None

            if new_items[u-1] > 0: # if the user is an active user, change both training and validation
                
                # new training row
                new_training_dict[self.uid_field] = new_training_dict.get(self.uid_field, []) + [u] #user id
                new_training_dict[self.iid_field] = new_training_dict.get(self.iid_field, []) + [int(g[self.iid_field].values)] #item id

                if timestamp is not None:
                    new_training_dict[self.time_field] = new_training_dict.get(self.time_field, []) + [timestamp] #timestamp
                    
                # new validation row
                new_valid_dict[self.uid_field] = new_valid_dict.get(self.uid_field, []) + [u] #user id
                new_valid_dict[self.iid_field] = new_valid_dict.get(self.iid_field, []) + [int(new_items[u-1])] #item id

                if timestamp is not None:
                    new_valid_dict[self.time_field] = new_valid_dict.get(self.time_field, []) + [timestamp+1] #timestamp


            else: # user is not active, copy the same interaction again in the new valid
                # new validation row
                new_valid_dict[self.uid_field] = new_valid_dict.get(self.uid_field, []) + [u] #user id
                new_valid_dict[self.iid_field] = new_valid_dict.get(self.iid_field, []) + [int(g[self.iid_field].values)] #item id

                if timestamp is not None:
                    new_valid_dict[self.time_field] = new_valid_dict.get(self.time_field, []) + [timestamp+1] #timestamp
                    

        self.validation_set._dataset.inter_feat = Interaction(new_valid_dict) # set new valid
        self.validation_set._dataset.inter_feat.sort([self.uid_field, self.time_field])
        # translate current training set and old validation set in dataframe
        training_df = pd.DataFrame(self.training_set._dataset.inter_feat.numpy())
        new_train = pd.concat([training_df, pd.DataFrame(new_training_dict)], axis=0).reset_index(drop=True)
        self.training_set._dataset.inter_feat = Interaction(new_train.copy(deep = True))   
        self.training_set._dataset.inter_feat.sort([self.uid_field, self.time_field]) 

        torch.cuda.empty_cache()




    # Hyperparameter tuning for the model. Perform a random search for
    # tuning the hyperparameter for the model
    # @input hyper_file (string): name of the file with the values of the hyperparameter
    def tuning(self, hyper_file):

        def objective_function(params_dict=None, config_file_list=None):

            if self.config['seed'] is not None and self.config['reproducibility'] is not None:
                init_seed(self.config['seed'], self.config['reproducibility'])

            dataset = create_dataset(self.config)
            train_data, valid_data, test_data = data_preparation(self.config, dataset)
            model_name = self.config['model']
            model = get_model(model_name)(self.config, train_data._dataset).to(self.config['device'])
            trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)

            return {
                'model': model_name,
                'best_valid_score': best_valid_score,
                'valid_score_bigger': self.config['valid_metric_bigger'],
                'best_valid_result': best_valid_result,
                'test_result': test_result
            }

        hp = HyperTuning(objective_function=objective_function, algo='random', early_stop=10,
                    max_evals=100, params_file=hyper_file, fixed_config_file_list=['environment.yaml'], params_dict=self.config_dict)
        
        print('starting tuning phase -------')
        hp.run()
        self.config_dict.update(hp.best_params)

        # checking no tuple, need all list
        for k in self.config_dict.keys():
            if isinstance(self.config_dict[k], tuple):
                self.config_dict[k] = list(self.config_dict[k])

        self.config = Config(config_file_list=['environment.yaml'], config_dict=self.config_dict)


        print('ended tuning phase ----')



    def compute_metrics(self, recommended_items):

        

        # distinct items proposed (collective)
        self.metrics['L_col'] = self.metrics.get('L_col', []) + [len(set(recommended_items.flatten())) -1] # -1 is padding value, not considered

        # radius of gyration (individual)
        rog = metrics.compute_rog(self.training_set._dataset)['radius_of_gyration']
        self.metrics['rog_ind'] = self.metrics.get('rog_ind', []) + [np.mean(rog)]
        # radius of gyration k = 2 (individual)
        k_rog = metrics.compute_rog(self.training_set._dataset, k = 2)['radius_of_gyration']
        self.metrics['rog_ind_2'] = self.metrics.get('rog_ind_2', []) + [np.mean(k_rog)]
       
        # distinct items for each user (individual)
        di = metrics.distinct_items(self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['D_ind'] = self.metrics.get('D_ind', []) + [np.mean(di)]

        # old items suggested (individual)
        old = metrics.old_items_suggested(recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['L_old_ind'] = self.metrics.get('L_old_ind', []) + [np.mean(old)]
        
        # new items suggested (individual)
        new = metrics.old_items_suggested(recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['L_new_ind'] = self.metrics.get('L_new_ind', []) + [np.mean(new)]
        # mean entropy (individual)
        entropy = metrics.uncorrelated_entropy(pd.DataFrame(self.training_set._dataset.inter_feat.cpu().numpy()), self.uid_field, self.iid_field)
        self.metrics['S_ind'] = self.metrics.get('S_ind', []) + [np.mean(entropy['entropy'])]

        # mean entropy (collective)
        entropy = metrics.uncorrelated_entropy(pd.DataFrame(self.training_set._dataset.inter_feat.cpu().numpy()), self.iid_field, self.uid_field)['entropy']
        self.metrics['S_col'] = self.metrics.get('S_col', []) + [np.mean(entropy)]

        # explore and return events (individual)
        explore, returns = metrics.get_explore_returns(self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['Expl_ind'] = self.metrics.get('Expl_ind', []) + [np.mean(explore)]
        self.metrics['Ret_ind'] = self.metrics.get('Ret_ind', []) + [np.mean(returns)]

        # individual gini index
        gini_ind = metrics.individual_gini(self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['Gini_ind'] = self.metrics.get('Gini_ind', []) + [np.mean(gini_ind)]



    def generate_not_rec_predictions(self):

        np.random.seed()

        users = list(self.training_set._dataset.user_counter.keys())
        m = self.training_set.dataset.inter_matrix().toarray()[users, :]

        match self.not_rec:
            case 'ir': # individual random
                def choose_ind_random(interaction):
                    return np.random.choice(interaction.nonzero()[0], 10)
                
                random_list = np.apply_along_axis(choose_ind_random, 1, m) # select 10 non-zero items from the history (ind. for each user)

                return  np.apply_along_axis(np.random.choice, 1, random_list)# choose one item per user from his history
            
            case 'cr': # collective random
                # get all unique items
                items = list(self.training_set._dataset.item_counter.keys())
                return np.random.choice(items, len(users)) # choose one item per user
            

            case 'ip': # individual popularity
                ind_pop_items = np.apply_along_axis(np.argsort, 1, m) # top-k items for each user
                return np.apply_along_axis(np.random.choice, 1, ind_pop_items) # choose one item per user


            case 'cp': # collective popularity (most popular items)
                # sort item counter and get the first 10
                pop_items = sorted(self.training_set._dataset.item_counter, key = self.training_set._dataset.item_counter.get, reverse=True)[:10]
                return np.random.choice(pop_items, len(users)) # choose one item per user
            
            case _:
                raise NotImplementedError
