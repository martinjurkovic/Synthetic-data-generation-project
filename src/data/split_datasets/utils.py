import os
import pandas as pd
from sklearn.model_selection import KFold


def read_tables(dataset_name, split_by="-", name_index=-1, **kwargs):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/original/' + dataset_name + '/'
    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_name = file[:-4].split(split_by)[name_index]
            table = pd.read_csv(
                path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
            tables[table_name] = table
    return tables


def save_metadata(metadata, dataset_name):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/metadata'

    metadata.to_json(f'{path}/{dataset_name}_metadata.json')


def split_k_fold(df, n_splits=10, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    for train_index, test_index in kf.split(df):
        train_fold = df.iloc[train_index]
        test_fold = df.iloc[test_index]
        folds.append((train_fold, test_fold))
    return folds

def split_on_parent(parent_folds, child_table, left_on, right_on):
    child_folds = []
    for i, (parent_train, parent_test) in enumerate(parent_folds):
        child_train = pd.merge(parent_train, child_table, how="right", left_on=left_on, right_on=right_on)
        child_test = pd.merge(parent_test, child_table, how="right", left_on=left_on, right_on=right_on)

        child_folds.append((child_train, child_test))

    return child_folds

def split_k_fold_on_parent(parent_folds, child_table, split_col_names):
    child_folds = []
    for i, (parent_train, parent_test) in enumerate(parent_folds):
        # get indexes of rows in child table which have molecule_id in parent_train
        child_train_indexes = []
        child_test_indexes = []
        for idx, (split_child, split_parent)  in enumerate(split_col_names):
            if idx == 0:
                child_train_indexes = child_table[child_table[split_child].isin(parent_train[split_parent])].index
                child_test_indexes = child_table[child_table[split_child].isin(parent_test[split_parent])].index
            else:
                # keep only indexes in child_train_indexes of rows in child table which have molecule_id in parent_train
                child_train_indexes = child_table[child_table[split_child].isin(parent_train[split_parent])].index.intersection(child_train_indexes)
                child_test_indexes = child_table[child_table[split_child].isin(parent_test[split_parent])].index.intersection(child_test_indexes)

        child_train = child_table.iloc[child_train_indexes]
        child_test = child_table.iloc[child_test_indexes]
        child_folds.append((child_train, child_test))
    return child_folds

def split_k_fold_on_multiple_parents(parents_folds, child_table, split_col_names):
    """
    split_col_names: 2D array of split_col_names pairs
    
    """
    child_folds = []
    for i, parent_folds in enumerate(parents_folds):
        if i == 0:
            child_folds = split_k_fold_on_parent(parent_folds, child_table, split_col_names[i])
        else:
            temp_folds = split_k_fold_on_parent(parent_folds, child_table, split_col_names[i])
            for j, (child_train, child_test) in enumerate(temp_folds):
                # get an intersection of indexes of child_train and child_folds[j][0]
                child_train_indexes = child_train.index.intersection(child_folds[j][0].index)
                child_test_indexes = child_test.index.intersection(child_folds[j][1].index)
                child_folds[j] = (child_folds[j][0].loc[child_train_indexes], child_folds[j][1].loc[child_test_indexes])

