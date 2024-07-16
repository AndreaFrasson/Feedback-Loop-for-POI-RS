from Feedback_Loop import FeedBack_Loop
import os


if __name__ =='__main__':
    parameter_dict = {
        'model' : 'Pop',
        'dataset':'foursquare',
        'data_path': os.getcwd(),
        'USER_ID_FIELD': 'uid',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['uid', 'item_id', 'timestamp']},
        'field_separator': ',',
        'benchmark_filename': ['part1', 'part2', 'part3'],
        'epochs': 20,
        'eval_args': {
            'group_by': 'user',
            'order': 'TO', # temporal order
            'split': {'LS': 'valid_and_test'}, # leave one out
            'mode': 'uni1'
        },
        'eval_batch_size': 4096
    }

    fl = FeedBack_Loop(parameter_dict)
    fl.loop(20,2)