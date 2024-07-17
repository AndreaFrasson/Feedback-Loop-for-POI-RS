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


class FeedBack_Loop():

    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.config = Config(config_file_list=['environment.yaml'], config_dict = config_dict)


    # first creation of the dataset and set the id variables. Split in training, validation and test 
    # with the configuration settings
    def initialize(self):
        self.dataset = create_dataset(self.config)

        self.iid_field = self.dataset.iid_field
        self.uid_field = self.dataset.uid_field
        self.time_field = self.dataset.time_field

        self.training_set, self.validation_set, self.test_set = data_preparation(self.config, self.dataset)
    

    # Main Loop 
    # if not specified, the model uses default values, otherwise before going in the loop it performs
    # a random search to tune the hyperparameters (dataset splitted in train-val-test). 
    # Then, the dataset is prepared and splitted and the loop starts. 
    def loop(self, epochs, dtrain, choice = 'r', tuning = False, hyper_file = None, user_frac = 1):

        self.epochs = epochs
        self.dtrain = dtrain
        self.metrics = {}

        if tuning:
            if not isinstance(hyper_file, str):
                raise NotImplementedError
            
            self.tuning(hyper_file)

        self.initialize()

        print(self.config['device'])

        for c in tqdm(range(self.epochs)):
            
            #extract user that will see the recommendations
            if user_frac < 1:
                user_acc = np.random.choice(list(self.training_set._dataset.user_counter.keys()),
                                            int(len(list(self.training_set._dataset.user_counter.keys())) * user_frac)) 
                user_acc -=1
            else:
                user_acc = None

            # every delta_train epochs, retrain of the model
            if c % self.dtrain == 0:
                # get model
                self.model = get_model(self.config['model'])(self.config, self.training_set._dataset).to(self.config['device'])
                # trainer loading and initialization
                self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
                # model training
                best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
                results = self.trainer.evaluate(self.test_set)
                self.metrics['test_hit'] = self.metrics.get('test_hit', []) + [results['hit@10']]
                self.metrics['test_precision'] = self.metrics.get('test_precision', []) + [results['precision@10']]
                self.metrics['test_rec'] = self.metrics.get('test_rec', []) + [results['recall@10']]


            predictions = self.generate_prediction(self.training_set._dataset, user_acc)
            # choose one item
            chosen_items = utils.choose_item(predictions, self.training_set._dataset, choice)

            if c % self.dtrain == 0:
                self.compute_metrics()

            self.update_incremental(chosen_items)
            


    # perform a fit step
    def fit(self):
        best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
        return best_valid_score, best_valid_result
    

    # evaluate the model using the test set
    def evaluate(self):
        results = self.trainer.evaluate(self.test_set)
        return results
    


    def generate_prediction(self, dataset, user_acc = None):

        torch.cuda.empty_cache()
        scores = self.__prediction()

        if user_acc is not None:
            scores[user_acc] = -1

        return scores
    

    # given a list of users, a matrix of items visited, and the model make the prediction for each user.
    # @return only the 10 best items, INTERNAL EMBEDDING, predicted for each user
    def __prediction(self):

        users = len(self.training_set._dataset.user_counter.keys())

        try:
            input_inter = Interaction({
                'uid': torch.tensor(list(self.training_set._dataset.user_counter.keys())),
                'iid': self.training_set._dataset.inter_feat[self.uid_field].reshape(users,-1),
                'timestamp': torch.tensor(np.unique(self.training_set._dataset.inter_feat['timestamp']))
            })
        
        except:
            input_inter = Interaction({
                'uid': torch.tensor(list(self.training_set._dataset.user_counter.keys())),
                'iid': self.training_set._dataset.inter_feat[self.uid_field].reshape(users,-1),
            })


        
        input_inter.to(self.model.device)
        print(input_inter.device)

        with torch.no_grad():
            try:  # if model have full sort predict
                scores = self.model.full_sort_predict(input_inter).cpu().reshape((users, -1))
            except NotImplementedError:  # if model do not have full sort predict
                len_input_inter = len(input_inter)
                input_inter = input_inter.repeat(self.training_set._dataset.item_num)
                input_inter.update(self.training_set._dataset.get_item_feature().repeat(len_input_inter))  # join item feature
                scores = self.model.predict(input_inter)
            
            scores = scores.view(-1, self.dataset.item_num)
        
        # get the 10 items with highest scores
        #rec_list = np.argsort(scores, axis = 1)[:, -10:]
        topk_score, topk_iid_list  = torch.topk(scores, 10)

        return topk_iid_list.numpy().flatten().reshape(-1,10)
    



    # Update the interaction files (for training) in an incremental fashion.
    def update_incremental(self, new_items):

        # translate current training set and validation set in dataframe
        training_df = pd.DataFrame(self.training_set._dataset.inter_feat.numpy())
        validation_df = pd.DataFrame(self.validation_set._dataset.inter_feat.numpy())

        # append the old validation set to the old training set, the timestamp increases by one
        try:
            new_train = pd.concat([training_df, validation_df]).sort_values(by = [self.uid_field, self.time_field])
        except:
            new_train = pd.concat([training_df, validation_df]).sort_values(by = [self.uid_field])
        # update the training set
        self.training_set._dataset.inter_feat = Interaction(new_train.copy(deep = True))
        #save file
        #new_train.columns = [self.uid_field+':token', self.iid_field+':token', self.time_field+':token']
        #new_train.to_csv(self.config_dict['data_path']+'/'+self.config_dict['dataset']+'/'+self.config_dict['dataset']+'.part1.inter', index=False)

        # update the timestamp for the new validation set
        valid_user = self.validation_set._dataset.inter_feat['uid'].cpu().numpy()

        try:
            new_timestamp = self.validation_set._dataset.inter_feat['timestamp'].cpu().numpy() + 1
            
            # the new validation set is build from the recommendetion made in the last step
            new_valid = pd.DataFrame(list(zip(valid_user, new_items, new_timestamp)), 
                        columns=[self.uid_field, self.iid_field, self.time_field])
        
        except:
            # the new validation set is build from the recommendetion made in the last step
            new_valid = pd.DataFrame(list(zip(valid_user, new_items)), 
                        columns=[self.uid_field, self.iid_field])
        
        # update the validation set
        self.validation_set._dataset.inter_feat = Interaction(new_valid.copy(deep=True))
        #save file
        #new_valid.columns = [self.uid_field+':token', self.iid_field+':token', self.time_field+':token']
        #new_valid.to_csv(self.config_dict['data_path']+'/'+self.config_dict['dataset']+'/'+self.config_dict['dataset']+'.part2.inter', index=False)
        
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



    def compute_metrics(self):

        recommended_items = self.generate_prediction(self.training_set._dataset)

        # distinct items proposed (collective)
        self.metrics['L_col'] = self.metrics.get('L_col', []) + [len(set(recommended_items.flatten()))]

        # radius of gyration (individual)
        self.metrics['rog_ind'] = self.metrics.get('rog_ind', []) + [metrics.compute_rog(self.training_set._dataset)['radius_of_gyration'].mean()]
        # radius of gyration k = 2 (individual)
        self.metrics['rog_ind_2'] = self.metrics.get('rog_ind_2', []) + [metrics.compute_rog(self.training_set._dataset, k = 2)['radius_of_gyration'].mean()]
       
        # distinct items for each user (individual)
        self.metrics['D_ind'] = self.metrics.get('D_ind', []) + [np.mean(metrics.distinct_items(self.training_set._dataset, self.uid_field, self.iid_field))]

        # old items suggested (individual)
        self.metrics['L_old_ind'] = self.metrics.get('L_old_ind', []) + [metrics.old_items_suggested(
                                                                           recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)]
        
        # new items suggested (individual)
        self.metrics['L_new_ind'] = self.metrics.get('L_new_ind', []) + [metrics.new_items_suggested(
                                                                           recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)]
        # mean entropy (individual)
        entropy = metrics.uncorrelated_entropy(pd.DataFrame(self.training_set._dataset.inter_feat.cpu().numpy()), self.uid_field, self.iid_field)
        self.metrics['S_ind'] = self.metrics.get('S_ind', []) + [np.mean(entropy['entropy'])]

        # mean entropy (collective)
        entropy = metrics.uncorrelated_entropy(pd.DataFrame(self.training_set._dataset.inter_feat.cpu().numpy()), self.iid_field, self.uid_field)
        self.metrics['S_col'] = self.metrics.get('S_col', []) + [np.mean(entropy['entropy'])]

        # explore and return events (individual)
        explore, returns = metrics.get_explore_returns(self.training_set._dataset, self.uid_field, self.iid_field)
        self.metrics['Expl_ind'] = self.metrics.get('Expl_ind', []) + [np.mean(explore)]
        self.metrics['Ret_ind'] = self.metrics.get('Ret_ind', []) + [np.mean(returns)]

        # individual gini index
        self.metrics['Gini_ind'] = self.metrics.get('Gini_ind', []) + [np.mean(metrics.individual_gini(self.training_set._dataset, self.uid_field, self.iid_field))]
