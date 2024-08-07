import torch
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import Pop
from recbole.data import Interaction
import numpy as np

from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances


class iCF(Pop):
    """Random is an fundamental self that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset, N_u = 17):
        self.dataset = dataset
        super(iCF, self).__init__(config, dataset)
        self.N_u = N_u
        


    def predict(self, interaction):
        return torch.rand(len(interaction)).squeeze(-1)


    def full_sort_predict(self, interaction, dataset = None):

        train_users = torch.unique(self.dataset.inter_feat[self.USER_ID]).reshape(-1,1)
        m_train = sparse.csr_matrix(self.dataset.inter_matrix().T)

        items = np.array(list(self.dataset.item_counter.keys())).reshape(-1,1)

        inter_users = torch.unique(interaction[self.USER_ID]).reshape(-1,1)

        if dataset is not None:
            m_inter = sparse.csr_matrix(dataset.inter_matrix().T)
        else:
            m_inter = sparse.csr_matrix(self.dataset.inter_matrix().T)

        users = torch.unique(interaction[self.USER_ID]).reshape(-1,1)

        #m = (m / m.sum(1)) # normalize row by total interactions

        # average interactions for all items
        avg_int_train = np.array((m_train.sum(1) / m_train.astype(bool).sum(axis=1)).flatten())
        avg_int_inter = np.array((m_inter.sum(1) / m_inter.astype(bool).sum(axis=1)).flatten())

        avg_int_train = np.nan_to_num(avg_int_train, 0)
        avg_int_inter = np.nan_to_num(avg_int_inter, 0)

        # similarity matrix between each item in the test with each item in the train
        sim_mat = 1 - pairwise_distances(m_inter, m_train, 'cosine')
        sim_mat = np.nan_to_num(sim_mat, 0)

        #sim_mat = torch.Tensor(sim_mat).to(self.device)

        # most similar items for item in the test set
        neighbors = np.argsort(sim_mat, 1)[:, -self.N_u:]

        def get_pred_cf(arr, m, avg_int,sim_mat, neighbors):
            item = int(arr.item())
            ne = neighbors[item]

            ws = m[ne] - avg_int.reshape(-1,1)[ne]

            print(ws)

            num = np.sum(sim_mat[item, ne] * ws, 0)
            den = np.sum(np.abs(sim_mat[item,ne]))

            scores = avg_int.reshape(-1,1)[item] + (num/den)
            np.nan_to_num(scores, 0)

            return scores


        pred = np.apply_along_axis(get_pred_cf, 1, arr = np.array(range(neighbors.shape[0])).reshape(-1,1), m = m_train, 
                                   avg_int = avg_int_inter, sim_mat = sim_mat, neighbors = neighbors)
        pred = pred.T
        pred = np.nan_to_num(pred, 0)
        return torch.tensor(np.argsort(pred, 1)[users.flatten(), -10:] + 1)
    

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