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

    def __init__(self, config_dict, steps):
        self.config_dict = config_dict
        self.config = Config(config_file_list=['environment.yaml'], config_dict = config_dict)
        self.steps = steps 



    def initialize(self):
        self.dataset = create_dataset(self.config)

        self.iid_field = self.dataset.iid_field
        self.uid_field = self.dataset.uid_field
        self.time_field = self.dataset.time_field

        self.training_set, self.validation_set, self.test_set = data_preparation(self.config, self.dataset)
    


    def loop(self, MaxIt, choice = 'r', tuning = False, hyper_file = None):
        self.metrics = {}
        if tuning:
            if not isinstance(hyper_file, str):
                raise NotImplementedError
            
            self.tuning(hyper_file)

        self.initialize()

        for c in tqdm(range(MaxIt)):
            if c % self.steps == 0:
                # get model
                self.model = get_model(self.config['model'])(self.config, self.training_set.dataset).to(self.config['device'])
                # trainer loading and initialization
                self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
                # model training
                best_valid_score, best_valid_result = self.trainer.fit(self.training_set, self.validation_set)
                results = self.trainer.evaluate(self.test_set)
                self.metrics['test_hit'] = self.metrics.get('test_hit', []) + [results['hit@10']]
                self.metrics['test_precision'] = self.metrics.get('test_precision', []) + [results['precision@10']]

            predictions, external_ids = self.generate_prediction(self.training_set._dataset)
            # choose one item
            chosen_tokens, chosen_ids = utils.choose_item(external_ids, self.training_set._dataset, choice)

            self.update_incremental(chosen_ids)
            self.compute_metrics(predictions)
            


    
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
        
        return rec_list, np.apply_along_axis(utils._from_ids_to_int, 1, external_item_list)
    



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
        self.config = Config(config_file_list=['environment.yaml'], config_dict=self.config_dict)
        print('ended tuning phase ----')


    def compute_metrics(self, recommended_items):
        self.metrics['card'] = self.metrics.get('card', []) + [len(set(recommended_items.flatten().numpy()))]
        self.metrics['rog_ind'] = self.metrics.get('rog_ind', []) + [metrics.compute_rog(self.training_set._dataset)['radius_of_gyration'].mean()]
        self.metrics['rog_ind_2'] = self.metrics.get('rog_ind_2', []) + [metrics.compute_rog(self.training_set._dataset, k = 2)['radius_of_gyration'].mean()]
        #self.metrics['D_ind'] = self.metrics.get('D_ind', []) + [metrics.distinct_items(self.training_set._dataset)]
        self.metrics['L_old_ind'] = self.metrics.get('L_old_ind', []) + [metrics.old_items_suggested(
                                                                           recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)]
        self.metrics['L_new_ind'] = self.metrics.get('L_new_ind', []) + [metrics.new_items_suggested(
                                                                           recommended_items, self.training_set._dataset, self.uid_field, self.iid_field)]
        entropy = metrics.uncorrelated_entropy(pd.DataFrame(self.training_set._dataset.inter_feat.cpu().numpy()), self.uid_field, self.iid_field)
        self.metrics['S_ind'] = self.metrics.get('S_ind', []) + [np.mean(entropy['entropy'])]