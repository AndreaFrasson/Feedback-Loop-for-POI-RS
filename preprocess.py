import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import pickle
import os


def preprocess():
    path = os.getcwd() + '/data/foursquare_complete.csv'
    # open dataset
    foursquare = pd.read_csv('data/foursquare_complete.csv')

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
    #timestamp
    red_df['timestamp'] = np.arange(0, min_len).tolist() * len(set_uid)

    # encoding ids
    enc = LabelEncoder()
    enc.fit(red_df['venue_id'])
    red_df['venue_id'] = enc.transform(red_df['venue_id'])

    mapping_id = dict(zip(range(len(enc.classes_)), enc.classes_))
    with open('id_category.pkl', 'wb') as f:
        pickle.dump(mapping_id, f)

    ### split train/test dataset
    users = set(red_df['uid'])

    red_df.columns = ['uid:token', 'venue_id:token', 'venue_category_name:token', 'lat:float', 'lon:float', 'timestamp:token']

    #interaction
    red_df[['uid:token', 'venue_id:token', 'timestamp:token']].to_csv('foursquare/foursquare.inter', index = False, sep = '\t')
    #users
    pd.DataFrame(set(red_df['uid:token']), columns=['uid:token']).to_csv('foursquare/foursquare.user', index=False, sep = '\t')
    #items
    items = red_df[['venue_id:token', 'venue_category_name:token']].drop_duplicates()
    items.sort_values(by = 'venue_id:token', inplace=True)
    items.to_csv('foursquare/foursquare.item', index = False, sep = '\t')