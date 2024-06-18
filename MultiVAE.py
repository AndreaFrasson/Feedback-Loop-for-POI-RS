from preprocess import preprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from Feedback_Loop import FeedBack_Loop

# SETTINGS
MODEL = 'MultiVAE'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 30
DEVICE_ID = '0'

# Default parameters
LEARNING_RATE = 0.005



if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 3:
        m = 5
        MaxIt = 20
    else:
        m = int(sys.argv[1])
        MaxIt = int(sys.argv[2])

        # make the atomic files form the data
    seed = 1234 # to get always the same users in train/test
    preprocess(seed)


    config_dict = {
            'model': MODEL,
            'data_path': DATA_PATH,
            'top_k': TOP_K,
            'dataset': DATASET,
            'epochs': EPOCHS,
            'use_gpu': len(DEVICE_ID) > 0,
            'device_id': DEVICE_ID,
            'learnign_rate': LEARNING_RATE,
        }


    fl = FeedBack_Loop(config_dict, m)
    fl.loop(MaxIt, 'r', True, 'MultiVAE.hyper')

    # all plot in the same picture
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 7))
    # generates a converging sequence for heavy ball
    iterations = [i for i in range(len(fl.metrics['L_col']))]
    training_step = [i for i in np.arange(len(fl.metrics['test_hit']))]

    axs[0,0].set_title('Diversity of Items')
    axs[0,0].plot(iterations, fl.metrics['L_col'], color = 'blue', linestyle = 'dashed')
    axs[0,0].set_xticks(range(len(iterations)))
    axs[0,0].vlines(np.arange(len(iterations), step=m), ymin=min(fl.metrics['L_col']), ymax= max(fl.metrics['L_col']), colors='red',linestyles='dotted')


    axs[0,1].set_title('Test metrics')
    axs[0,1].plot(training_step, fl.metrics['test_hit'])
    axs[0,1].set_xticks(range(len(training_step)))
    axs[0,1].legend(['Hit@10',])


    axs[0,2].set_title('New item proposed')
    axs[0,2].plot(iterations, fl.metrics['L_new_ind'], color = 'blue', linestyle = 'dashed')
    axs[0,2].set_xticks(range(len(iterations)))
    axs[0,2].vlines(np.arange(len(iterations), step=m), ymin=min(fl.metrics['L_new_ind']), ymax= max(fl.metrics['L_new_ind']), colors='red',linestyles='dotted')


    axs[1,0].set_title('ROG')
    axs[1,0].plot(iterations, fl.metrics['rog_ind'], color = 'blue', linestyle = 'dashed')
    axs[1,0].set_xticks(range(len(iterations)))
    axs[1,0].vlines(np.arange(len(iterations), step=m), ymin=min(fl.metrics['rog_ind']), ymax= max(fl.metrics['rog_ind']), colors='red',linestyles='dotted')


    axs[1,1].set_title('S_col')
    axs[1,1].plot(iterations, fl.metrics['S_col'], color = 'blue', linestyle = 'dashed')
    axs[1,1].set_xticks(range(len(iterations)))
    axs[1,1].vlines(np.arange(len(iterations), step=m), ymin=min(fl.metrics['S_col']), ymax= max(fl.metrics['S_col']), colors='red',linestyles='dotted')


    axs[1,2].set_title('Gini_ind')
    axs[1,2].plot(iterations, fl.metrics['Gini_ind'], color = 'blue', linestyle = 'dashed')
    axs[1,2].set_xticks(range(len(iterations)))
    axs[1,2].vlines(np.arange(len(iterations), step=m), ymin=min(fl.metrics['Gini_ind']), ymax= max(fl.metrics['Gini_ind']), colors='red',linestyles='dotted')
    # Adjust layout
    plt.tight_layout()
    # Show plots
    plt.savefig('plot/run_MultiVAE_TrStep_'+str(m)+'_.png')
