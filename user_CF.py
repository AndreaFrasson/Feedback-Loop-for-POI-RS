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


class uCF(Pop):
    """Random is an fundamental self that recommends random items."""

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
            m = sparse.csr_matrix(self.dataset.inter_matrix())
        else:
            m = sparse.csr_matrix(dataset.inter_matrix())
        
        m = normalize(m, norm = 'l1', copy = True)

        # average interactions for all users
        avg_int = np.array((m.sum(1) / m.astype(bool).sum(axis=1)).flatten())
        sim_mat = squareform(pdist(m.todense(), 'cosine'))
        sim_mat = np.nan_to_num(sim_mat, False)

        # neghbors for each user
        N_u = 12
        neighbors = np.argsort(sim_mat, 1)[:, -N_u:]

        def get_pred_cf(user, m, avg_int,sim_mat, neighbors):
            user = int(user.item())
            j = np.where(m[user].toarray() == 0)[1]
            ne = neighbors[user]

            int_ne = m[:, j][ne]

            # compute the weighted sum between sim(u_a, u_k)*(m_k,j - r_k), but only for the users who have rated an item
            weighted_sum = int_ne.toarray().astype(bool) * np.array(m[:, j][ne].toarray() - avg_int.reshape(-1,1)[ne])

            num = sim_mat[user,ne].reshape(-1,1) * weighted_sum
            #num = sim_mat[user,ne].reshape(-1,1) * np.array(m[ne].toarray() - avg_int.reshape(-1,1)[ne])

            num = np.sum(num, axis = 0)
            den = np.dot(sim_mat[user,ne].reshape(1,-1), int_ne.toarray().astype(bool)) # sum, for each item, only the sim of the users that interacted with the item

            scores = avg_int.reshape(-1,1)[user] + (num/den)
            scores = np.nan_to_num(scores, False).flatten()

            pos = np.argsort(scores)[-10:]
            return j[pos]

        pred = np.apply_along_axis(get_pred_cf, 1, arr = users, m = m, avg_int = avg_int, sim_mat = sim_mat, neighbors = neighbors)
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