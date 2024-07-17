from Feedback_Loop import FeedBack_Loop

from recbole.quick_start import run_recbole
import os


if __name__ =='__main__':

    run_recbole(model='NFM', dataset='ml-100k')