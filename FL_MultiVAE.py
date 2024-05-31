import utils as u
from preprocess import preprocess
import metrics
import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import data_preparation, create_dataset
from recbole.quick_start.quick_start import get_model, get_trainer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# SETTINGS
MODEL = 'MultiVAE'
DATA_PATH = os.getcwd() 
TOP_K = 10
DATASET = 'foursquare'
EPOCHS = 10

# Default parameters
LEARNING_RATE = 0.005


def run_BPR(default = False):

    # make the atomic files form the data
    seed = 1234 # to get always the same users in train/test
    preprocess(seed)

    # train users
    train_users = pd.read_csv('foursquare/foursquare.part1.inter', sep = ',')['uid:token'].to_list()
    train_users = list(set(train_users))

    config_dict = {
            'model': MODEL,
            'data_path': DATA_PATH,
            'top_k': TOP_K,
            'dataset': DATASET,
            'epochs': EPOCHS
        }

    if default:
        tuned_params = {
            'learnign_rate': LEARNING_RATE,
        }
    else:
        #hyperparameter tuning
        tuned_params = u.tuning('BPR', 'MultiVAE.hyper', config_dict)
    
    config_dict.update(tuned_params)
    # create the configurator
    config = Config(config_file_list=['environment.yaml'], config_dict = config_dict)

    # environment settings dor the loop
    m = 3
    MaxIt = 20
    c = 0

    # output metrics
    hit = []
    prec = []
    card = []
    mean_entropy_train = []

    # Main Loop
    for c in tqdm(range(MaxIt), smoothing=1):

        if c % m == 0:  
            # create the dataset
            dataset = create_dataset(config)
            # split
            training_set, validation_set, test_set = data_preparation(config, dataset)

            # get model
            model = get_model(config['model'])(config, training_set.dataset).to(config['device'])

            # trainer loading and initialization
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
            # model training
            best_valid_score, best_valid_result = trainer.fit(training_set, validation_set)
            results = trainer.evaluate(test_set)
            hit.append(results['hit@10'])
            prec.append(results['precision@10'])

        # generate synthetic data for the training
        uid_series = training_set._dataset.token2id(training_set._dataset.uid_field, [str(s) for s in train_users])
        items = training_set._dataset.inter_feat[training_set._dataset.iid_field].reshape(len(train_users), -1)
        rec_list = u.prediction(uid_series, items, model)

        # translate the location id back to the original embedding
        external_item_list = training_set.dataset.id2token(training_set.dataset.iid_field, rec_list.cpu())
        card.append(len(np.unique(external_item_list.flatten())))

        # choose one item
        chosen_items = u.choose_item(rec_list, training_set._dataset, 'c')

        # update the training/validation file
        u.update_incremental(chosen_items, training_set, validation_set)

        # compute the entropy for the new training set
        entropy_train = metrics.uncorrelated_entropy(pd.read_csv('foursquare/foursquare.part1.inter', sep = ','), 'uid:token', 'venue_id:token')
        # mean value for the users
        mean_entropy_train.append(np.mean(entropy_train['entropy'].to_numpy()))

    return card, hit, prec, mean_entropy_train


if __name__ == '__main__':
    card, hit, prec, mean_entropy = run_BPR()

    # all plot in the same picture
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    # generates a converging sequence for heavy ball
    iterations = [i for i in range(len(card))]
    training_step = [i for i in np.arange(len(hit))]

    axs[0].set_title('Diversity of Items')
    axs[0].plot(iterations, card, color = 'blue', linestyle = 'dashed')
    axs[0].set_xticks(range(len(iterations)))
    axs[0].vlines(np.arange(len(iterations), step=3), ymin=min(card), ymax= max(card), colors='red',linestyles='dotted')

    axs[1].set_title('Test metrics')
    axs[1].plot(training_step, hit)
    axs[1].plot(training_step, prec)
    axs[1].set_xticks(range(len(training_step)))
    axs[1].legend(['Hit@10', 'Precision@10'])

    axs[2].set_title('Mean Normalized Entropy')
    axs[2].plot(iterations, mean_entropy, color = 'blue', linestyle = 'dashed')
    axs[2].set_xticks(range(len(iterations)))
    axs[2].vlines(np.arange(len(iterations), step=3), ymin=min(mean_entropy), ymax= max(mean_entropy), colors='red',linestyles='dotted')
    # Adjust layout
    plt.tight_layout()
    # Show plots
    plt.show()