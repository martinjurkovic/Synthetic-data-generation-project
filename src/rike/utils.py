import os
import pandas as pd
import re

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'


def read_tables(dataset_name, leave_out_fold_num, type, split_by="_", name_index=1, limit=None, synthetic=False, method_name=None, **kwargs):
    highest_fold = 99999
    if limit is not None:
        highest_fold = limit

    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    if synthetic:
        path = cwd_project + '/data/synthetic/' + dataset_name + '/' + method_name + '/'
    else:
        path = cwd_project + '/data/splits/' + dataset_name + '/'

    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_split = file[:-4].split(split_by)
            table_name = table_split[name_index]
            for i in range(name_index + 1, len(table_split)):
                if table_split[i] not in ("fold"):
                    table_name += "_" + table_split[i]
                else:
                    break

            fold = file[:-4].split(split_by)[-1]
            if fold == str(leave_out_fold_num) and type == "test":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                tables[table_name] = table
            elif fold != str(leave_out_fold_num) and type == "train" and int(fold) < highest_fold:
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if table_name not in tables:
                    tables[table_name] = table
                else:
                    tables[table_name] = pd.concat([tables[table_name], table])
    return tables


def get_train_test_split(dataset_name, leave_out_fold_num, limit=None, synthetic=False, method_name = None):
    tables_train = read_tables(dataset_name, leave_out_fold_num, "train", limit=limit, synthetic=synthetic, method_name=method_name)
    tables_test = read_tables(dataset_name, leave_out_fold_num, "test", limit=limit, synthetic=synthetic, method_name=method_name)
    return tables_train, tables_test


def save_train_test_split(dataset_name, leave_out_fold_num, tables_train, tables_test):
    path = CWD_PROJECT + '/data/splits/' + dataset_name + '/' + dataset_name + '_leave_out_' + str(leave_out_fold_num)
    # create directory if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    for table_name, table in tables_train.items():
        table.to_csv(path + f'{"/" if path[-1] != "/" else ""}' + table_name + "_train_" + str(leave_out_fold_num) + ".csv", index=False)
    for table_name, table in tables_test.items():
        table.to_csv(path + f'{"/" if path[-1] != "/" else ""}' + table_name + "_test_" + str(leave_out_fold_num) + ".csv", index=False)


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


def ends_with_digit(string):
    pattern = r"_train_\d$"
    return bool(re.search(pattern, string))


def save_data(tables_synthetic, dataset_name, leave_out_fold_num, method='SDV', index=False):
    path = CWD_PROJECT + '/data/synthetic/' + dataset_name + '/' + method + '/'
    # create directory if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    for table_name, table in tables_synthetic.items():
        if ends_with_digit(table_name):
            table_name = table_name[:-8]
        table.to_csv(
            path + f'{dataset_name}_{table_name}_fold_{leave_out_fold_num}.csv', index=index)


def find_fk(parent, reference, metadata):
    for field in metadata.to_dict()['tables'][parent]['fields']:
        if field in reference:
            return field
    return None 


def merge_children(tables, metadata, root):
    parent = tables[root]
    children = metadata.get_children(root)
    for child in children:
        fks = metadata.get_foreign_keys(root, child)
        for fk in fks:
            parent_fk = find_fk(root, fk, metadata)
            if parent_fk is None:
                continue
            child_table = merge_children(tables, metadata, child)
            if fk in parent.columns:
                parent = parent.merge(child_table, left_on=fk, right_on=fk, how='outer', suffixes=(f'_{root}', f'_{child}'))
            else:
                # this happens when there are 2 foreign keys from the same table
                # e.g. bond in biodegradabaility with fks atom_id and atom_id_2
                parent = parent.merge(child_table, left_on=parent_fk, right_on=fk, how='outer', suffixes=(f'_{root}', f'_{child}')) 
    return parent


def conditionally_sample(tables, metadata, root):
    parent = tables[root]
    children = metadata.get_children(root)
    for child in children:
        child_table = tables[child]
        fks = metadata.get_foreign_keys(root, child)
        for fk in fks:
            parent_fk = find_fk(root, fk, metadata)
            if parent_fk is None:
                continue
            parent_ids = parent[parent_fk].unique()
            child_table = child_table[child_table[fk].isin(parent_ids)]
            tables[child] = child_table
            tables = conditionally_sample(tables, metadata, child)
    return tables

