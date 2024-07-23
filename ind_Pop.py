import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop

class ind_Pop(Pop):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        self.dataset = dataset
        super(ind_Pop, self).__init__(config, dataset)
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction):
        users = torch.unique(interaction[self.USER_ID])

        m = self.dataset.inter_matrix().toarray()

        pop_results = np.apply_along_axis(np.argsort, 1, m)[1:users[-1]+1, -10:]

        return torch.tensor(pop_results).reshape((-1, 10))
    

    def evaluate(self, test_interaction, dataset):
        users = torch.unique(test_interaction[self.USER_ID])

        k = 10 # number of prediction
        results = torch.tensor([]).to(self.device)
        
        for i in range(k):
            results = torch.cat([results, self.full_sort_predict(test_interaction, dataset)], dim = 1)
        
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
    

    