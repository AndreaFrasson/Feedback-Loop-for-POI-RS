import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop
import numpy as np
from scipy.spatial.distance import pdist, squareform


class CF(Pop):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        self.dataset = dataset
        super(CF, self).__init__(config, dataset)
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction, dataset = None):

        users = torch.unique(interaction[self.USER_ID])

        #similarity between users
        if dataset is None:
            m = self.dataset.inter_matrix().toarray()[users]
        else:
            m = dataset.inter_matrix().toarray()[users]

        sim_mat = squareform(pdist(m, metric='cosine'))
        np.fill_diagonal(sim_mat, 0)

        def __len_flatnonzero(a):
            return len(np.flatnonzero(a))

        # average interactions for all users
        avg_int = np.sum(m, axis = 1) / np.apply_along_axis(__len_flatnonzero, 1, m)

        # neghbors for each user
        N_u = 3
        neighbors = np.argsort(sim_mat)[:, -N_u:]


        def get_pred_cf(u, m, avg_int,sim_mat, neighbors):
            u = int(u.item())
            ne = neighbors[u-1]

            j = np.where(m[u-1] == 0)[0][1:]

            num = np.sum(sim_mat[u-1,ne].reshape(-1,1) * (m[:, j][ne] - avg_int[ne].reshape(-1,1)), axis = 0)
            den = np.sum(sim_mat[u-1,ne])

            scores = num/den
            pos = np.argsort(scores)[-10:]
            pred = j[pos]

            return pred


        pred = np.apply_along_axis(get_pred_cf, 1, arr = users, m = m, avg_int = avg_int, sim_mat = sim_mat, neighbors = neighbors)

        return pred
    

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