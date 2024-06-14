import utils as u
from preprocess import preprocess
import metrics
import pandas as pd
import numpy as np
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer
import os
from tqdm import tqdm
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


def run(m = 3, MaxIt = 20, default = False):

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




if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    if n < 3:
        m = 5
        MaxIt = 20
    else:
        m = int(sys.argv[1])
        MaxIt = int(sys.argv[2])

    run(m, MaxIt)

    pd.DataFrame(card).to_csv('card.csv', index = False)
    pd.DataFrame(hit).to_csv('hit.csv', index = False)
    pd.DataFrame(prec).to_csv('prec.csv', index = False)

    # all plot in the same picture
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    # generates a converging sequence for heavy ball
    iterations = [i for i in range(len(card))]
    training_step = [i for i in np.arange(len(hit))]

    axs[0].set_title('Diversity of Items')
    axs[0].plot(iterations, card, color = 'blue', linestyle = 'dashed')
    axs[0].set_xticks(range(len(iterations)))
    axs[0].vlines(np.arange(len(iterations), step=m), ymin=min(card), ymax= max(card), colors='red',linestyles='dotted')

    axs[1].set_title('Test metrics')
    axs[1].plot(training_step, hit)
    axs[1].plot(training_step, prec)
    axs[1].set_xticks(range(len(training_step)))
    axs[1].legend(['Hit@10', 'Precision@10'])

    axs[2].set_title('Mean Normalized Entropy')
    axs[2].plot(iterations, mean_entropy, color = 'blue', linestyle = 'dashed')
    axs[2].set_xticks(range(len(iterations)))
    axs[2].vlines(np.arange(len(iterations), step=m), ymin=min(mean_entropy), ymax= max(mean_entropy), colors='red',linestyles='dotted')
    # Adjust layout
    plt.tight_layout()
    # Show plots
    plt.savefig('plot/run_MultiVAE_TrStep_'+str(m)+'_.png')