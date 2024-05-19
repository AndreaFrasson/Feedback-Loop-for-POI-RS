import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import pickle
import os


def og_atomic_files():
    path = os.getcwd() + '/data/foursquare_complete.csv'
    # open dataset
    foursquare = pd.read_csv(path)
    #foursquare['geometry'] = list(zip(foursquare['lat'], foursquare['lon']))

    # encoding category
    enc = LabelEncoder()
    enc.fit(foursquare['venue_category_name'])
    foursquare['venue_category_name'] = enc.transform(foursquare['venue_category_name'])

    #save dictionary of the categories
    mapping = dict(zip(range(len(enc.classes_)), enc.classes_))
    with open('name_category.pkl', 'wb') as f:
        pickle.dump(mapping, f)


    #equal length in all of the trajectory, cut to min length
    min_len = float('inf')
    set_uid = set(foursquare['uid'])
    for u in set_uid:
        min_len = min(min_len, len(foursquare[foursquare['uid'] == u]))

    new_df = []
    for u in set_uid:
        to_append = foursquare[foursquare['uid'] == u].iloc[:min_len, :].values.tolist()
        for r in to_append:
            new_df.append(r)

    new_df = pd.DataFrame(new_df, columns=foursquare.columns)
    new_df['uid'] = new_df['uid'].astype(int)

    #timestamp
    new_df['timestamp'] = np.arange(1, 101).tolist() * len(set_uid)

    #inter file for recbole
    red_df = new_df[['uid', 'venue_id', 'timestamp', 'venue_category_name']].copy()
    red_df.columns = ('uid:token', 'venue_id:token', 'timestamp:token', 'venue_category_name:token')

    # encoding ids
    enc = LabelEncoder()
    enc.fit(red_df['venue_id:token'])
    red_df['venue_id:token'] = enc.transform(red_df['venue_id:token'])

    mapping = dict(zip(range(len(enc.classes_)), enc.classes_))
    with open('id_category.pkl', 'wb') as f:
        pickle.dump(mapping, f)

    # interaction file
    red_df[['uid:token', 'venue_id:token', 'timestamp:token']].to_csv('foursquare/foursquare.inter', index=False, sep = '\t')
    #item file for recbole
    items = red_df[['venue_id:token', 'venue_category_name:token']].drop_duplicates().sort_values(by=['venue_id:token'])
    items.to_csv('foursquare/foursquare.item', index = False, sep='\t')
    #user file
    pd.DataFrame(set(red_df['uid:token']), columns=['uid:token']).to_csv('foursquare/foursquare.user', index=False, sep = '\t')