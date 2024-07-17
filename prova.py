from Feedback_Loop import FeedBack_Loop

from recbole.quick_start import run_recbole
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


if __name__ =='__main__':
    # dataset config : Context-aware Recommendation
    config = {'load_col':{
        'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
        'user': ['user_id', 'age', 'gender', 'occupation'],
        'item': ['item_id', 'release_year', 'class']},
        'threshold': {'rating': 4},
        'normalize_all': True}
    # configurations initialization
    config = Config(model='FNM', config_dict=config, dataset='ml-100k')

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)
    run_recbole(model='NFM', dataset='ml-100k')