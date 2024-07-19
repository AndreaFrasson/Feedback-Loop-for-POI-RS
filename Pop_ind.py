from recbole.model.general_recommender import Pop
from recbole.utils import InputType, ModelType
import torch


class Pop_ind(Pop):
    r"""Pop is an fundamental model that always recommend the most popular item."""
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL


    def __init__(self, config, dataset):
        super(Pop_ind, self).__init__(config, dataset)

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

        self.max_cnt = torch.max(self.item_cnt, dim=1)[0]

        return torch.nn.Parameter(torch.zeros(1)).to(self.device)
    
    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        result = torch.true_divide(self.item_cnt[item, :], self.max_cnt)
        return result.squeeze(-1)

    