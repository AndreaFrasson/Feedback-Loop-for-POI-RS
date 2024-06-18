import pandas as pd
import numpy as np
from scipy import stats
from skmob.measures.individual import radius_of_gyration
import skmob
import utils


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


def topk_history(dataframe, k = 10):
        d_index = []

        topk_items = dataframe['venue_id'].value_counts().index[:k]

        for idx, row in dataframe.iterrows():
            if row['venue_id'] not in topk_items:
                d_index.append(idx)

        return d_index



def compute_rog(interactions, k = None):

    pois_df = pd.read_csv('foursquare/foursquare.item')
    pois_df.columns = ['venue_id', 'category', 'lat', 'lon']

    iid_field = interactions.iid_field
    uid_field = interactions.uid_field

    users = utils._from_ids_to_int(interactions.id2token(interactions.uid_field, interactions.inter_feat[uid_field]))
    items = utils._from_ids_to_int(interactions.id2token(interactions.iid_field, interactions.inter_feat[iid_field]))

    interactions = pd.DataFrame({'uid':users, 'venue_id':items})

    # get the trajectory dataframe
    tdf = skmob.TrajDataFrame(interactions.merge(pois_df[['venue_id', 'lat', 'lon']], how = 'left'), latitude='lat', longitude='lon', user_id='uid')

    if k is None: # compute the global radius of gyrations
        rog = radius_of_gyration(tdf, False)
        return rog
    
    if k == 0:
        print('k can\'t be 0')
        return
    
    # keep only the k most visited locations, for each user

    # find the indexes to drop
    drop_list = tdf.groupby('uid').apply(topk_history, k = k)
    # drop the 'rarest' locations
    tdf = tdf.drop([item for row in drop_list for item in row])

    #compute the ROG
    rog_k = radius_of_gyration(tdf, False)

    return rog_k



# count, for each user, how many items in the recommendetion list proposed
# were already saw. Return the sum
def old_items_suggested(recommended_items, dataset, uid_field, iid_field):
    users = set(dataset.inter_feat[uid_field].numpy())
    history = dataset.inter_feat[iid_field].numpy().reshape(len(users), -1)

    old_items = 0
    for rec, hist in list(zip(recommended_items, history)):
        listC =[list(set(hist)).count(x) for x in rec]
        old_items += sum(listC)

    return old_items



# count, for each user, how many items in the recommendetion list proposed
# are new Return the sum
def new_items_suggested(recommended_items, dataset, uid_field, iid_field):
    users = set(dataset.inter_feat[uid_field].numpy())
    history = dataset.inter_feat[iid_field].numpy().reshape(len(users), -1)

    new_items = 0
    for rec, hist in list(zip(recommended_items, history)):
        listC =[list(set(hist)).count(x) for x in rec]
        new_items += len(rec) - sum(listC)

    return new_items



# for each user, compute the number of distinct items in his/her history
def distinct_items(dataset, uid_field, iid_field):
    df = pd.DataFrame(dataset.inter_feat.numpy())
    return df.groupby(uid_field)[iid_field].nunique()



#può essere calcolato all'inizio e poi aggiornato
def _explore_return(traj):
    explore = set()
    returns = 0
    for l in traj:
        if l not in explore:
            explore.add(l)
        else:
            returns+=1
    return len(explore), returns



def get_explore_returns(dataset, uid_field, iid_field):
    df = pd.DataFrame(dataset.inter_feat.numpy())
    expl, ret = zip(*df.groupby(uid_field)[iid_field].apply(_explore_return).to_numpy())
    return expl, ret



def _gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))



def RelativeFrequency(my_list): 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    return np.array(list(freq.values())) / len(freq.values())



def individual_gini(dataset, uid_field, iid_field):
    df = pd.DataFrame(dataset.inter_feat.numpy())
    return df.groupby(uid_field)[iid_field].apply(RelativeFrequency).apply(_gini)


