# Created by mqgao at 2018/11/23

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""

import pandas as pd
import os


def get_model_path(station): return 'model/{}-model.model'.format(station)


def get_train_dataset_path(station):
    path = 'train-dataset/point_date_' + station + '.csv'

    if not os.path.exists(path):
        raise NameError(
            'The Train Data for {} is not exist, please check it!'.format(station)
        )

    else:
        return path


def format_predicate_to_pd(datetime, predicate):
    dataframe = {
        'time': list(datetime),
        'predication': list(predicate),
    }

    return pd.DataFrame.from_dict(dataframe)
