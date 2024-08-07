import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop

class ind_Random(Pop):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        self.dataset = dataset
        super(ind_Random, self).__init__(config, dataset)
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction):
        users = torch.unique(interaction[self.USER_ID])
        rand_results = []

        for u in users:
            try:
                history = interaction[interaction[self.USER_ID] == u][self.ITEM_ID]

            except:
                history = self.dataset.inter_feat[self.dataset.inter_feat[self.USER_ID] == u][self.ITEM_ID]
            idx = np.random.randint(0,len(history))
            rand_results.append(int(history[idx]))

        return torch.tensor(rand_results).reshape(-1,1)
    

    def evaluate(self, test_interaction):
        users = torch.unique(test_interaction[self.USER_ID])

        k = 10 # numbero of prediction
        results = torch.tensor([]).to(self.device)
        for i in range(k):
            results = torch.cat([results, self.full_sort_predict(test_interaction)], dim = 1)
        
        # compute metrics and return a dict
        y = []
        for u in users:
            y.append(int(test_interaction[test_interaction[self.USER_ID] == u][self.ITEM_ID][-1]))
        
        
        hit_sum = 0 
        prec_sum = 0
        rec_sum = 0

        for i in range(len(y)):
            hit_sum += int(y[i] in results[i,:])
            intersection = len(set(results[i,:].numpy()).intersection(set([y[i]])))
            prec_sum +=  intersection / len([y[0]])
            rec_sum += intersection / len(results[i,:])

        results = {
            'hit@10': hit_sum/len(users),
            'precision@10': prec_sum/len(users),
            'recall@10': rec_sum/len(users)
        }

        return results