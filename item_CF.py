import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop
from recbole.data import Interaction
import numpy as np
from scipy.spatial.distance import pdist, squareform

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class iCF(Pop):
    """Random is an fundamental self that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        self.dataset = dataset
        super(iCF, self).__init__(config, dataset)
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction, dataset = None):

        users = torch.unique(interaction[self.USER_ID]).reshape(-1,1)
        if dataset is None:
            m = sparse.csr_matrix(self.dataset.inter_matrix().T)
        else:
            m = sparse.csr_matrix(dataset.inter_matrix().T)

        m = (m / m.sum(1)) # normalize row by total interactions

        # average interactions for all users
        avg_int = np.array((m.sum(1) / m.astype(bool).sum(axis=1)).flatten())
        np.nan_to_num(avg_int, 0)

        sim_mat = cosine_similarity(m)
        np.fill_diagonal(sim_mat, 0)

        # neghbors for each user
        N_u = 7
        neighbors = np.argsort(sim_mat, 1)[:, -N_u:]

        def get_pred_cf(user, m, avg_int,sim_mat, neighbors):
            user = int(user.item())
            j = np.where(m[user] == 0)[0]
            ne = neighbors[user]

            # compute the weighted sum between sim(u_a, u_k)*(m_k,j - r_k), but only for the users who have rated an item
            ws = (m[ne] - np.nan_to_num(avg_int.reshape(-1,1))[ne])
            #[:, j]

            num = np.sum(sim_mat[user][ne].reshape(-1,1) * ws, 0)
            den = sum(sim_mat[user][ne])

            scores = avg_int.reshape(-1,1)[user] + (num/den)
            np.nan_to_num(scores, 0)

            pos = np.argsort(scores)[-10:]# return the item not visited with highest scores
            return pos

        pred = np.apply_along_axis(get_pred_cf, 1, arr = users, m = m.toarray(), avg_int = avg_int, sim_mat = sim_mat, neighbors = neighbors)
        return torch.tensor(pred)
    

    def evaluate(self, dataset):
        users = torch.unique(dataset.inter_feat[self.USER_ID])

        prediction = self.full_sort_predict(Interaction({self.USER_ID: users}), dataset)

        y = []
        for u in users:
            y.append(int(dataset.inter_feat[dataset.inter_feat[self.USER_ID] == u][self.ITEM_ID][-1]))


        # compute metrics and return a dict
        hit_sum = 0 
        prec_sum = 0
        rec_sum = 0

        for i in range(len(y)):
            hit_sum += int(y[i] in prediction[i,:])
            intersection = len(set(prediction[i,:]).intersection(set([y[i]])))
            prec_sum +=  intersection / len([y[0]])
            rec_sum += intersection / len(prediction[i,:])


        results = {
            'hit@10': hit_sum/len(users),
            'precision@10': prec_sum/len(users),
            'recall@10': rec_sum/len(users)
        }

        return results