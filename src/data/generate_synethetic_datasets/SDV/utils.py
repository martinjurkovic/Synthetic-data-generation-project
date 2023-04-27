import os
import pandas as pd

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'


def read_tables(dataset_name, leave_out_fold_num, type, split_by="_", name_index=1, **kwargs):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/splits/' + dataset_name + '/'
    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_name = file[:-4].split(split_by)[name_index]
            fold = file[:-4].split(split_by)[-1]
            if fold == str(leave_out_fold_num) and type == "test":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                tables[table_name] = table
            elif fold != str(leave_out_fold_num) and type == "train":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if table_name not in tables:
                    tables[table_name] = table
                else:
                    tables[table_name] = pd.concat([tables[table_name], table])
    return tables

def get_train_test_split(dataset_name, leave_out_fold_num):
    tables_train = read_tables(dataset_name, leave_out_fold_num, "train")
    tables_test = read_tables(dataset_name, leave_out_fold_num, "test")
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

def save_metadata(metadata, dataset_name):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/metadata'

    metadata.to_json(f'{path}/{dataset_name}_metadata.json')

def save_SDV_data(tables_synthetic, dataset_name, leave_out_fold_num):
    path = CWD_PROJECT + '/data/synthetic/' + dataset_name + '/'
    for table_name, table in tables_synthetic.items():
        table.to_csv(
            path + f'{dataset_name}_{table_name}-fold-{leave_out_fold_num}.csv', index=False)