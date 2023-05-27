import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from rike.generation import sdv_metadata
from rike.utils import get_train_test_split, read_original_tables, conditionally_sample, add_number_of_children, add_children_column_means
from rike.evaluation.metrics import ks_test, cardinality_similarity, xgboost_detection, calculate_statistical_metrics


def add_table(results, table_name, metric_type="single_table"):
    if table_name not in results["metrics"][metric_type].keys():
        results["metrics"][metric_type][table_name] = {}
    return results


def add_metric(results, metric_name, metric_value, k=0, table_name=None, metric_type="multi_table"):
    if metric_type == "multi_table":
        if metric_name not in results["metrics"]["multi_table"] or k == 0:
            results["metrics"]["multi_table"][metric_name] = {
                "scores": []}
        results["metrics"]["multi_table"][metric_name]["scores"].append(
            metric_value)
    elif metric_type == "single_table":
        if metric_name not in results["metrics"]["single_table"][table_name] or k == 0:
            results["metrics"]["single_table"][table_name][metric_name] = {
                "scores": []}
        results["metrics"]["single_table"][table_name][metric_name]["scores"].append(
            metric_value)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    return results


def aggregate_results(metrics, score_type="scores"):
    scores = metrics[score_type]
    metrics["mean"] = np.mean(scores, axis=0).tolist()
    metrics["std"] = np.std(scores, axis=0).tolist()
    metrics["min"] = np.min(scores, axis=0).tolist()
    metrics["max"] = np.max(scores, axis=0).tolist()
    metrics["median"] = np.median(scores, axis=0).tolist()
    metrics['se'] = (np.std(scores, axis=0) / np.sqrt(len(scores))).tolist()
    metrics['95ci'] = [[np.quantile(s, 0.025, axis=0), np.quantile(s, 0.975, axis=0)] for s in np.array(scores).T]
    return metrics

def set_zurich_type(tables_train):
    tables_train['claims']['customer_id'] = tables_train['claims']['customer_id'].astype(int, errors='ignore')
    tables_train['claims']['claim_id'] = tables_train['claims']['claim_id'].astype(int, errors='ignore')
    tables_train['claims']['policy_id'] = tables_train['claims']['policy_id'].astype(int, errors='ignore')
    tables_train['policies']['customer_id'] = tables_train['policies']['customer_id'].astype(int, errors='ignore')
    return tables_train


def generate_report(dataset_name, method_name, single_table_metrics=[xgboost_detection], multi_table_metrics = ['cardinality'], statistical_results = True, save_report=False, limit=10):
    # read metrics_report from file
    if os.path.exists(f"metrics_report/{dataset_name}_{method_name}.json"):
        with open(f"metrics_report/{dataset_name}_{method_name}.json", "r") as f:
            metrics_report = json.load(f)
    else:
        metrics_report = {
            "dataset_name": dataset_name,
            "method_name": method_name,
            "metrics": {
                "single_table": {
                },
                "multi_table": {
                }
            }
        }

    # load metadata
    # tables_original = read_original_tables(dataset_name)
    # if dataset_name in ("zurich_mle", "zurich"):
    #     tables_original = set_zurich_type(tables_original)
    # metadata = sdv_metadata.generate_metadata(dataset_name, tables_original)
    tables_train, tables_test = get_train_test_split(dataset_name, 0, limit=limit)
    if dataset_name == "zurich_mle":
        tables_train = tables_test.copy()
    if dataset_name in ("zurich_mle", "zurich"):
        tables_train['claims']['customer_id'] = tables_train['claims']['customer_id'].astype(int)
        tables_train['claims']['claim_id'] = tables_train['claims']['claim_id'].astype(int)
        tables_train['claims']['policy_id'] = tables_train['claims']['policy_id'].astype(int)
        tables_train['policies']['customer_id'] = tables_train['policies']['customer_id'].astype(int)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)

    for k in tqdm(range(min(10, limit))):
        tables_orig_train, tables_orig_test = get_train_test_split(
            dataset_name, leave_out_fold_num=k, synthetic=False, 
            method_name=method_name, limit=limit, metadata=metadata)
        tables_synthetic_train, tables_synthetic_test = get_train_test_split(
            dataset_name, leave_out_fold_num=k, synthetic=True, 
            method_name=method_name, limit=limit, metadata=metadata)
        # if dataset_name in ("zurich_mle", "zurich"):
        #     tables_synthetic_train = set_zurich_type(tables_synthetic_train)
        #     tables_synthetic_test = set_zurich_type(tables_synthetic_test)
        #     tables_orig_train = set_zurich_type(tables_orig_train)
        #     tables_orig_test = set_zurich_type(tables_orig_test)
        
        
        # select the same amout of rows for original and synthetic tables
        # based on the number of rows in the root table
        # TODO: refactor to a function
        root_table = sdv_metadata.get_root_table(dataset_name)
        # train
        nrows = min(tables_orig_train[root_table].shape[0], tables_synthetic_train[root_table].shape[0])
        tables_orig_train[root_table] = tables_orig_train[root_table].sample(n=nrows, random_state=0)
        tables_synthetic_train[root_table] = tables_synthetic_train[root_table].sample(n=nrows, random_state=0)
        tables_synthetic_train = conditionally_sample(tables_synthetic_train, metadata, root_table)
        tables_orig_train = conditionally_sample(tables_orig_train, metadata, root_table)
        # test
        nrows = min(tables_orig_test[root_table].shape[0], tables_synthetic_test[root_table].shape[0])
        tables_orig_test[root_table] = tables_orig_test[root_table].sample(n=nrows, random_state=0)
        tables_synthetic_test[root_table] = tables_synthetic_test[root_table].sample(n=nrows, random_state=0)
        tables_synthetic_test = conditionally_sample(tables_synthetic_test, metadata, root_table)
        tables_orig_test = conditionally_sample(tables_orig_test, metadata, root_table)
        # tables with number of children rows
        original_train_children = {}
        synthetic_train_children = {}
        original_test_children = {}
        synthetic_test_children = {}
        # tables with children column means
        original_train_means = {}
        synthetic_train_means = {}
        original_test_means = {}
        synthetic_test_means = {}
        # tables with number of children rows and children column means
        original_train_children_means = {}
        synthetic_train_children_means = {}
        original_test_children_means = {}
        synthetic_test_children_means = {}
        for table, fields in metadata.to_dict()['tables'].items():
            for field, values in fields['fields'].items():
                # convert the datetime columns to datetime type
                if values['type'] == 'datetime':
                    tables_orig_test[table][field] = pd.to_datetime(tables_orig_test[table][field], format=values['format'])
                    tables_synthetic_test[table][field] = pd.to_datetime(tables_synthetic_test[table][field], format=values['format'])
                    tables_orig_train[table][field] = pd.to_datetime(tables_orig_train[table][field], format=values['format'])
                    tables_synthetic_train[table][field] = pd.to_datetime(tables_synthetic_train[table][field], format=values['format'])
            # make sure the columns are in the same order
            orig_columns = tables_orig_test[table].columns
            tables_synthetic_test[table] = tables_synthetic_test[table][orig_columns]
            tables_synthetic_test[table].columns = orig_columns
            tables_synthetic_train[table] = tables_synthetic_train[table][orig_columns]
            tables_synthetic_train[table].columns = orig_columns

            # Add number of children to each table
            original_train_children[table] = add_number_of_children(table, metadata, tables_orig_train)
            synthetic_train_children[table] = add_number_of_children(table, metadata, tables_synthetic_train)
            original_test_children[table] = add_number_of_children(table, metadata, tables_orig_test)
            synthetic_test_children[table] = add_number_of_children(table, metadata, tables_synthetic_test)
            # add children column means to each table
            original_train_means[table] = add_children_column_means(table, metadata, tables_orig_train)
            synthetic_train_means[table] = add_children_column_means(table, metadata, tables_synthetic_train)
            original_test_means[table] = add_children_column_means(table, metadata, tables_orig_test)
            synthetic_test_means[table] = add_children_column_means(table, metadata, tables_synthetic_test)
            # add children column means to each table
            cols_to_use = original_train_means[table].columns.difference(original_train_children[table].columns)
            original_train_children_means[table] = pd.merge(original_train_children[table], 
                                                            original_train_means[table][cols_to_use], 
                                                            left_index=True, right_index=True)
            synthetic_train_children_means[table] = pd.merge(synthetic_train_children[table],
                                                            synthetic_train_means[table][cols_to_use], 
                                                            left_index=True, right_index=True)
            original_test_children_means[table] = pd.merge(original_test_children[table],
                                                            original_test_means[table][cols_to_use], 
                                                            left_index=True, right_index=True)
            synthetic_test_children_means[table] = pd.merge(synthetic_test_children[table],
                                                            synthetic_test_means[table][cols_to_use], 
                                                            left_index=True, right_index=True)
            
            
        # SDV cardinality shape similarity
        if 'cardinality' in multi_table_metrics:
            cardinality = cardinality_similarity(tables_orig_test,
                            tables_synthetic_test,
                            metadata)

        # Multi table metrics
        for metric in multi_table_metrics:
            if metric == 'cardinality':
                continue
            metric_name = metric.__name__
            metric_value = metric(tables_orig_test, tables_synthetic_test,
                                    original_train=tables_orig_train, 
                                    synthetic_train=tables_synthetic_train, 
                                    metadata = metadata,
                                    root_table= sdv_metadata.get_root_table(dataset_name),
                                    save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_name}/probabilities/multi_table_{k}.csv")
            metrics_report = add_metric(metrics_report, metric_name, metric_value, k = k, metric_type='multi_table')
            # add metric results with number of children rows
            metric_with_children = metric.__name__ + '_with_children'
            metric_value_with_children = metric(original_test_children, synthetic_test_children,
                                    original_train=original_train_children, 
                                    synthetic_train=synthetic_train_children, 
                                    metadata = metadata,
                                    root_table= sdv_metadata.get_root_table(dataset_name),
                                    save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_children}/probabilities/multi_table_{k}.csv")
            metrics_report = add_metric(metrics_report, metric_with_children, metric_value_with_children, k = k, metric_type='multi_table')
            # add metric results with children column means
            metric_with_means = metric.__name__ + '_with_means'
            metric_value_with_means = metric(original_test_means, synthetic_test_means,
                                    original_train=original_train_means, 
                                    synthetic_train=synthetic_train_means, 
                                    metadata = metadata,
                                    root_table= sdv_metadata.get_root_table(dataset_name),
                                    save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_means}/probabilities/multi_table_{k}.csv")
            metrics_report = add_metric(metrics_report, metric_with_means, metric_value_with_means, k = k, metric_type='multi_table')
            # add metric results with children column means and children rows
            metric_with_children_means = metric.__name__ + '_with_children_and_means'
            metric_value_with_children_means = metric(original_test_children_means, synthetic_test_children_means,
                                    original_train=original_train_children_means, 
                                    synthetic_train=synthetic_train_children_means, 
                                    metadata = metadata,
                                    root_table= sdv_metadata.get_root_table(dataset_name),
                                    save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_children_means}/probabilities/multi_table_{k}.csv")
            metrics_report = add_metric(metrics_report, metric_with_children_means, metric_value_with_children_means, k = k, metric_type='multi_table')

        # Remove foreign key columns
        for table, fields in metadata.to_dict()['tables'].items():
            for field, values in fields['fields'].items():
                if 'ref' in values.keys():
                    #print('DROPPING', field, 'FROM', table)
                    # remove foreign keys for each table
                    tables_orig_test[table].drop(columns=[field], inplace=True)
                    tables_synthetic_test[table].drop(columns=[field], inplace=True)
                    tables_orig_train[table].drop(columns=[field], inplace=True)
                    tables_synthetic_train[table].drop(columns=[field], inplace=True)
                    # remove foreign keys for each table with children
                    original_train_children[table].drop(columns=[field], inplace=True)
                    synthetic_train_children[table].drop(columns=[field], inplace=True)
                    original_test_children[table].drop(columns=[field], inplace=True)
                    synthetic_test_children[table].drop(columns=[field], inplace=True)
                    

        # Single table metrics
        for table_name in tables_orig_test.keys():
            metrics_report = add_table(metrics_report, table_name)
            for metric in single_table_metrics:
                metric_name = metric.__name__
                metric_value = metric(tables_orig_test[table_name], tables_synthetic_test[table_name],
                                      original_train=tables_orig_train[table_name], 
                                      synthetic_train=tables_synthetic_train[table_name], 
                                      metadata = metadata.to_dict()['tables'][table_name],
                                      save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_name}/probabilities/{table_name}_{k}.csv")
                metrics_report = add_metric(metrics_report, metric_name, metric_value, table_name=table_name, k = k, metric_type='single_table')
                
                if len(metadata.get_children(table_name)) > 0:
                    # add metric results with number of children rows
                    metric_with_children = metric.__name__ + '_with_children'
                    metric_value_with_children = metric(original_test_children[table_name], synthetic_test_children[table_name],
                                            original_train=original_train_children[table_name], 
                                            synthetic_train=synthetic_train_children[table_name], 
                                            metadata = metadata.to_dict()['tables'][table_name],
                                            save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_children}/probabilities/{table_name}_{k}.csv")
                    metrics_report = add_metric(metrics_report, metric_with_children, metric_value_with_children, table_name=table_name, k = k, metric_type='single_table')
                    # add metric results with children column means
                    metric_with_means = metric.__name__ + '_with_means'
                    metric_value_with_means = metric(original_test_means[table_name], synthetic_test_means[table_name],
                                            original_train=original_train_means[table_name], 
                                            synthetic_train=synthetic_train_means[table_name], 
                                            metadata = metadata.to_dict()['tables'][table_name],
                                            save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_means}/probabilities/{table_name}_{k}.csv")
                    metrics_report = add_metric(metrics_report, metric_with_means, metric_value_with_means, table_name=table_name, k = k, metric_type='single_table')
                    # add metric results with children column means and children rows
                    metric_with_children_means = metric.__name__ + '_with_children_and_means'
                    metric_value_with_children_means = metric(original_test_children_means[table_name], synthetic_test_children_means[table_name],
                                            original_train=original_train_children_means[table_name], 
                                            synthetic_train=synthetic_train_children_means[table_name], 
                                            metadata = metadata.to_dict()['tables'][table_name],
                                            save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_with_children_means}/probabilities/{table_name}_{k}.csv")
                    metrics_report = add_metric(metrics_report, metric_with_children_means, metric_value_with_children_means, table_name=table_name, k = k, metric_type='single_table')

                if statistical_results:
                    scores = calculate_statistical_metrics(tables_orig_test[table_name], 
                                                        tables_synthetic_test[table_name],
                                                        metadata = metadata.to_dict()['tables'][table_name])
                    metrics_report = add_table(metrics_report, table_name, metric_type='single_table')
                    metrics_report = add_metric(metrics_report, 'statistical', scores, table_name=table_name, k = k, metric_type='single_table')
                    if len(metadata.get_children(table_name)) > 0:
                        scores = calculate_statistical_metrics(original_test_children[table_name], 
                                                                synthetic_test_children[table_name],
                                                                metadata = metadata.to_dict()['tables'][table_name])
                        metrics_report = add_table(metrics_report, table_name, metric_type='single_table')
                        metrics_report = add_metric(metrics_report, 'statistical_with_children', scores, table_name=table_name, k = k, metric_type='single_table')

            if 'cardinality' in multi_table_metrics:
                for score in cardinality[table_name]:
                    metrics_report = add_metric(metrics_report, 'cardinality', score, table_name=table_name, k = k, metric_type='single_table')

            


    for table_name in metrics_report["metrics"]["single_table"].keys():
        for metric, metrics in metrics_report["metrics"]["single_table"][table_name].items():
            metrics = aggregate_results(metrics)

    for metric, metrics in metrics_report["metrics"]["multi_table"].items():
        metrics = aggregate_results(metrics)
            
    # for metric, metrics in metrics_report["metrics"]["statistical"].items():
    #     metrics = aggregate_results(metrics)
    
    if save_report:
        with open(f"metrics_report/{dataset_name}_{method_name}.json", "w") as f:
            json.dump(metrics_report, f, indent=4, sort_keys=True)
    
    return metrics_report
