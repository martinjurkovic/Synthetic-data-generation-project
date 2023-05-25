import os
import pandas as pd
import re

from rike.generation import sdv_metadata

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'

def get_highest_fold(dataset_name, method_name, evaluation=False):
    if not evaluation:
        path = os.path.join(CWD_PROJECT, 'data', 'splits', dataset_name)
    else:
        path = os.path.join(CWD_PROJECT, 'data', 'synthetic', dataset_name, method_name)
        
    highest_fold = -1
    if not os.path.exists(path):
        return 10
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_split = file[:-4].split("_fold_")
            fold = int(table_split[1])
            if fold > highest_fold:
                highest_fold = fold
    return highest_fold

def read_tables(dataset_name, 
                leave_out_fold_num, 
                type, 
                split_by="_", 
                name_index=1, 
                limit=None, 
                synthetic=False, 
                method_name=None, 
                metadata = None, 
                evaluation = False, 
                delta = 1, 
                add_indexes = True,
                **kwargs):
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
            # TODO: refactor to function
            if fold == str(leave_out_fold_num) and type == "test":
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if metadata is not None and add_indexes:
                    table = add_fold_index_to_keys(fold, table_name, table, metadata)
                tables[table_name] = table
            elif not evaluation and fold != str(leave_out_fold_num) and type == "train" and int(fold) < highest_fold:
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if metadata is not None and add_indexes:
                    table = add_fold_index_to_keys(fold, table_name, table, metadata)
                if table_name not in tables:
                    tables[table_name] = table
                else:
                    tables[table_name] = pd.concat([tables[table_name], table])
            elif evaluation and fold == (int(leave_out_fold_num) + delta) % highest_fold and type == "train" and int(fold) < highest_fold:
                table = pd.read_csv(
                    path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
                if metadata is not None and add_indexes:
                    table = add_fold_index_to_keys(fold, table_name, table, metadata)
                tables[table_name] = table

    # for every table in tables, drop duplicate rows
    for table_name, table in tables.items():
        tables[table_name] = table.drop_duplicates()
    return tables


def subsample(dataset_name, leave_out_fold_num, synthetic=False, limit=None):
    if limit is None:
        max_folds = 11
    else:
        max_folds = limit
    if synthetic:
        k = leave_out_fold_num
    else:
        k = (leave_out_fold_num + 1) % max_folds - 1
    #print('test ==', k)
    tables_test = read_tables(dataset_name, k, "test")
    # train: read every other fold
    tables_train = {}
    for i in range(k + 2, k + max_folds - 1, 2):
        fold = i % (max_folds - 1)
        if fold == leave_out_fold_num:
            break
        #print(f'Adding fold, {fold} to synthetic=={synthetic} train: loo:{leave_out_fold_num}')
        fold_tables = read_tables(dataset_name, fold, "test")
        for table_name, table in fold_tables.items():
            if table_name not in tables_train:
                tables_train[table_name] = table
            else:
                tables_train[table_name] = pd.concat([tables_train[table_name], table])
    
    return tables_train, tables_test


def get_train_test_split(dataset_name, leave_out_fold_num, limit=None, synthetic=False, method_name = None, metadata = None, evaluation = False, delta = 1):
    if method_name == 'subsample':
        return subsample(dataset_name, leave_out_fold_num, synthetic=synthetic, limit=limit)
    tables_train = read_tables(dataset_name, leave_out_fold_num, "train", limit=limit, synthetic=synthetic, method_name=method_name, metadata=metadata, evaluation=evaluation, delta=delta)
    tables_test = read_tables(dataset_name, leave_out_fold_num, "test", limit=limit, synthetic=synthetic, method_name=method_name, metadata=metadata, evaluation=evaluation, delta=delta)
    return tables_train, tables_test

def add_fold_index_to_keys(leave_out_fold_num, table_name, table, metadata):
    pk = metadata.get_primary_key(table_name)
    keys = [pk]
    for parent in metadata.get_parents(table_name):
        fks = metadata.get_foreign_keys(parent, table_name)
        keys += fks
    for key in keys:
        table[key] = table[key].astype(str) + "_" + str(leave_out_fold_num)
    return table


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


def add_number_of_children(table, metadata, tables):
    parent = tables[table].copy()
    children = metadata.get_children(table)
    for child in children:
        child_table = add_number_of_children(child, metadata, tables)
        fks = metadata.get_foreign_keys(table, child)
        child_pk = metadata.get_primary_key(child)
        for fk in fks:
            parent_fk = find_fk(table, fk, metadata)
            if parent_fk is None:
                continue
            # count number of children for each parent row
            num_children = child_table.groupby(fk).count()[child_pk]
            num_children = num_children.reset_index()
            num_children.columns = [fk, f'{child}_count']
            # join number of children to parent table
            parent = parent.merge(num_children, left_on=parent_fk, right_on=fk, how='left')
            if fk != parent_fk:
                parent = parent.drop(columns=fk)

            # aggregate the number of grandchildren
            for column in child_table.columns:
                if f'_count' in column:
                    # add sum of grandchildren
                    num_grandchildren = child_table.groupby(fk).sum(numeric_only=True)[column]
                    num_grandchildren = num_grandchildren.reset_index()
                    num_grandchildren.columns = [fk, f'grandchildren_sum_{column}']
                    parent = parent.merge(num_grandchildren, left_on=parent_fk, right_on=fk, how='left')
                    # add mean of grandchildren
                    mean_grandchildren = child_table.groupby(fk).mean(numeric_only=True)[column]
                    mean_grandchildren = mean_grandchildren.reset_index()
                    mean_grandchildren.columns = [fk, f'grandchildren_mean_{column}']
                    parent = parent.merge(mean_grandchildren, left_on=parent_fk, right_on=fk, how='left')
        
        # where the parent id does not match set to 0
        for column in parent.columns:
            if f'{child}_count' in column:
                parent[column] = parent[column].fillna(0)
    return parent

