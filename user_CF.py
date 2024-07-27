import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop
import numpy as np
from scipy.spatial.distance import pdist, squareform

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class uCF(Pop):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        self.dataset = dataset
        super(uCF, self).__init__(config, dataset)
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction, dataset = None):

        users = torch.unique(interaction[self.USER_ID]).reshape(-1,1)
        if dataset is None:
            m = sparse.csr_matrix(self.dataset.inter_matrix())[users.flatten(),:]
        else:
            m = sparse.csr_matrix(dataset.inter_matrix())[users.flatten(),:]

        # average interactions for all users
        avg_int = (m.sum(1) / m.astype(bool).sum(axis=1)).flatten()

        sim_mat = cosine_similarity(m)

        # neghbors for each user
        N_u = 3
        neighbors = np.argsort(sim_mat[:, 1:], 1)[:, -N_u:]

        def get_pred_cf(user, m, avg_int,sim_mat, neighbors):
            user = int(user.item())
            ne = neighbors[user-1]

            j = np.where(m[user-1].toarray() == 0)[1]


            num = sim_mat[user-1,ne].reshape(-1,1) * np.array(m[:, j][ne].toarray() - avg_int.reshape(-1,1)[ne])

            num = np.sum(num, axis = 0)
            den = np.sum(sim_mat[user-1,ne])

            scores = num/den
            pos = np.argsort(scores)[-10:]
            pred = j[pos]

            return pred

        pred = np.apply_along_axis(get_pred_cf, 1, arr = users, m = m, avg_int = avg_int, sim_mat = sim_mat, neighbors = neighbors)
        return pred
    

    def evaluate(self, dataset):
        users = torch.unique(dataset.inter_feat[self.USER_ID])

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