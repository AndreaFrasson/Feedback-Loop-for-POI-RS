import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import pickle
import os


def preprocess(seed = 1234):
    np.random.seed(seed)

    path = os.getcwd()
    # open dataset
    foursquare = pd.read_csv(path + '/data/foursquare_complete.csv')

    try:
        os.mkdir('foursquare')
    except:
        pass

    # encoding category
    enc = LabelEncoder()
    enc.fit(foursquare['venue_category_name'])
    foursquare['venue_category_name'] = enc.transform(foursquare['venue_category_name'])

    mapping_name = dict(zip(range(len(enc.classes_)), enc.classes_))
    with open('name_category.pkl', 'wb') as f:
        pickle.dump(mapping_name, f)

    # equal length in all of the trajectory
    min_len = float('inf')
    set_uid = set(foursquare['uid'])
    for u in set_uid:
        min_len = min(min_len, len(foursquare[foursquare['uid'] == u]))

    red_df = foursquare.groupby(by=['uid']).tail(min_len).copy()
    red_df.reset_index(inplace=True, drop=True)
    #timestamp
    red_df['timestamp'] = np.arange(0, min_len).tolist() * len(set_uid)

    # encoding ids
    enc = LabelEncoder()
    enc.fit(red_df['venue_id'])
    red_df['venue_id'] = enc.transform(red_df['venue_id'])

    mapping_id = dict(zip(range(len(enc.classes_)), enc.classes_))
    with open('id_loc.pkl', 'wb') as f:
        pickle.dump(mapping_id, f)

    ### split train/valid/test dataset
    users = list(set(red_df['uid']))
    training_ratio = 0.8
    red_df.columns = ['uid:token', 'venue_id:token', 'venue_category_name:token', 'lat:float', 'lon:float', 'timestamp:token']

    train_users = np.random.choice(users, int(len(users)*training_ratio), replace=False)

    # get the indexes for the train and the test set
    remained_inter = pd.Series(True, index=red_df.index)
    remained_inter &= red_df['uid:token'].isin(train_users)
    next_index = [remained_inter[remained_inter].index.values, remained_inter[~remained_inter].index.values]

    # get train and test
    train, test = [red_df.iloc[index].sort_values(by = ['uid:token', 'timestamp:token']) for index in next_index]

    validation = train.groupby('uid:token').tail(1)
    train = train.groupby('uid:token').head(min_len-1)

    #users
    pd.DataFrame(set(red_df['uid:token']), columns=['uid:token']).to_csv('foursquare/foursquare.user', index=False, sep = ',')
    #items
    items = red_df[['venue_id:token', 'venue_category_name:token', 'lat:float', 'lon:float']].drop_duplicates(subset=['venue_id:token', 'venue_category_name:token'])
    items.sort_values(by = 'venue_id:token', inplace=True)
    items.to_csv('foursquare/foursquare.item', index = False, sep = ',')

    mapping_cat = dict(zip(red_df['venue_id:token'], red_df['venue_category_name:token']))
    with open('id_category.pkl', 'wb') as f:
        pickle.dump(mapping_cat, f)

    #interaction
    train[['uid:token', 'venue_id:token', 'timestamp:token']].to_csv('foursquare/foursquare.part1.inter', index = False, sep = ',')
    validation[['uid:token', 'venue_id:token', 'timestamp:token']].to_csv('foursquare/foursquare.part2.inter', index = False, sep = ',')
    test[['uid:token', 'venue_id:token', 'timestamp:token']].to_csv('foursquare/foursquare.part3.inter', index = False, sep = ',')

    

if __name__ == '__main__':
    preprocess()

