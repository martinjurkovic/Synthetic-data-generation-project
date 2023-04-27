import os
import pandas as pd

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'


def read_tables(dataset_name, test_fold_index, type, synthetic, method_name=None, split_by="_", name_index=1, **kwargs):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    mid_path = ''
    if synthetic:
        mid_path = '/data/synthetic/'
        path = cwd_project + '/data/synthetic/' + dataset_name + '/' + method_name + '/'
    else:
        path = cwd_project + '/data/splits/' + dataset_name + '/'
    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_name = file[:-4].split(split_by)[name_index]
            fold = file[:-4].split(split_by)[-1]
            if fold == str(test_fold_index) and type == "test":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                tables[table_name] = table
            elif fold != str(test_fold_index) and type == "train":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if table_name not in tables:
                    tables[table_name] = table
                else:
                    tables[table_name] = pd.concat([tables[table_name], table])
    return tables


def get_train_test_split(dataset_name, test_fold_index, synthetic=False, method_name = None):
    tables_train = read_tables(dataset_name, test_fold_index, "train", synthetic, method_name)
    tables_test = read_tables(dataset_name, test_fold_index, "test", synthetic, method_name)
    return tables_train, tables_test


def read_original_tables(dataset_name, split_by="-", name_index=-1, **kwargs):
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