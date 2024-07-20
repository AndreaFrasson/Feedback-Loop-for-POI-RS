from recbole.model.general_recommender import Pop
from recbole.utils import InputType, ModelType
import torch
import pandas as pd


class Pop_ind(Pop):
    r"""Pop is an fundamental model that always recommend the most popular item."""
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL


    def __init__(self, config, dataset):
        super(Pop_ind, self).__init__(config, dataset)

        self.item_cnt = torch.zeros(
                self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False
            )

        self.user_item_cnt = torch.zeros(
                self.n_users, self.n_items, dtype=torch.long, device=self.device, requires_grad=False
            )
        self.max_cnt = None


    def calculate_loss(self, interaction):

        df_inter = pd.DataFrame(interaction.numpy())
        for u,g in df_inter.groupby(self.USER_ID):
            item = torch.zeros(self.n_items, dtype=torch.long, device=self.device, requires_grad=False)
            history = g[self.ITEM_ID]
            for i in history:
                item[i] += 1

            self.user_item_cnt[u,] = item

        self.max_cnt = torch.max(self.user_item_cnt, dim=1)[0]

        return torch.nn.Parameter(torch.zeros(1)).to(self.device)
    
    

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        result = torch.true_divide(self.user_item_cnt, self.max_cnt.reshape(-1,1))
        return result.squeeze(-1)

    