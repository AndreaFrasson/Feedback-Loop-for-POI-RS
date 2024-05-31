import pandas as pd
import numpy as np
import sys
import torch
from scipy import stats

from tqdm import tqdm, tqdm_notebook
from recbole.data.interaction import Interaction


def _uncorrelated_entropy_individual(interactions, normalize=True):
    """
    Compute the uncorrelated entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    n = len(interactions)
    probs = interactions.value_counts(normalize=True).sort_index().to_numpy()
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(interactions.values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def uncorrelated_entropy(interactions, group_attr, value_attr):
    df = interactions.groupby(group_attr).apply(lambda x: _uncorrelated_entropy_individual(x[value_attr]))
    return pd.DataFrame(df).reset_index().rename(columns={0: 'entropy'})