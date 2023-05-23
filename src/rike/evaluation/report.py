import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from rike.generation import sdv_metadata
from rike.utils import get_train_test_split, read_original_tables, conditionally_sample, add_number_of_children
from rike.evaluation.metrics import ks_test, cardinality_similarity


def add_table(results, table_name):
    if table_name not in results["metrics"]["single_table"].keys():
        results["metrics"]["single_table"][table_name] = {}
    return results


def add_metric(results, metric_name, metric_value, k=0, table_name= None, multi_table=False):
    if multi_table:
        if metric_name not in results["metrics"]["multi_table"] or k == 0:
            results["metrics"]["multi_table"][metric_name] = {
                "scores": []}
        results["metrics"]["multi_table"][metric_name]["scores"].append(
            metric_value)
    else:
        if metric_name not in results["metrics"]["single_table"][table_name] or k == 0:
            results["metrics"]["single_table"][table_name][metric_name] = {
                "scores": []}
        results["metrics"]["single_table"][table_name][metric_name]["scores"].append(
            metric_value)
    return results



def generate_report(dataset_name, method_name, single_table_metrics=[ks_test], multi_table_metrics = ['cardinality'], save_report=False, limit=10):
    # read metrics_report from file
    if os.path.exists(f"metrics_report/{dataset_name}_{method_name}.json"):
        with open(f"metrics_report/{dataset_name}_{method_name}.json", "r") as f:
            metrics_report = json.load(f)
    else:
        metrics_report = {
            "dataset_name": dataset_name,
            "metrics": {
                "single_table": {
                },
                "multi_table": {
                },
            }
        }

    # load metadata
    tables_original = read_original_tables(dataset_name)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_original)

    for k in tqdm(range(min(10, limit))):
        tables_orig_train, tables_orig_test = get_train_test_split(
            dataset_name, leave_out_fold_num=k, synthetic=False, limit=limit)
        tables_synthetic_train, tables_synthetic_test = get_train_test_split(
            dataset_name, leave_out_fold_num=k, synthetic=True, method_name=method_name, limit=limit)
        
        
        # select the same amout of rows for original and synthetic tables
        # based on the number of rows in the root table
        root_table = sdv_metadata.get_root_table(dataset_name)
        # train
        nrows = min(tables_orig_test[root_table].shape[0], tables_synthetic_test[root_table].shape[0])
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
            
        original_train_children = {}
        synthetic_train_children = {}
        original_test_children = {}
        synthetic_test_children = {}
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
        # Add number of children to metadata


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
            metrics_report = add_metric(metrics_report, metric_name, metric_value, k = k, multi_table=True)

            if len(metadata.get_children(table_name)) > 0:
                metric_with_children = metric.__name__ + '_with_children'
                metric_value_with_children = metric(original_test_children, synthetic_test_children,
                                        original_train=original_train_children, 
                                        synthetic_train=synthetic_train_children, 
                                        metadata = metadata,
                                        root_table= sdv_metadata.get_root_table(dataset_name),
                                        save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_name}/probabilities/multi_table_{k}.csv")
                metrics_report = add_metric(metrics_report, metric_with_children, metric_value_with_children, k = k, multi_table=True)

        # Remove foreign key columns
        for table, fields in metadata.to_dict()['tables'].items():
            for field, values in fields['fields'].items():
                if 'ref' in values.keys():
                    tables_orig_test[table].drop(columns=[field], inplace=True)
                    tables_synthetic_test[table].drop(columns=[field], inplace=True)
                    tables_orig_train[table].drop(columns=[field], inplace=True)
                    tables_synthetic_train[table].drop(columns=[field], inplace=True)

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
                metrics_report = add_metric(metrics_report, metric_name, metric_value, table_name=table_name, k = k)

                if len(metadata.get_children(table_name)) > 0:
                    metric_with_children = metric.__name__ + '_with_children'
                    metric_value_with_children = metric(original_test_children[table_name], synthetic_test_children[table_name],
                                            original_train=original_train_children[table_name], 
                                            synthetic_train=synthetic_train_children[table_name], 
                                            metadata = metadata.to_dict()['tables'][table_name],
                                            save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_name}/probabilities/{table_name}_{k}.csv")
                    metrics_report = add_metric(metrics_report, metric_with_children, metric_value_with_children, table_name=table_name, k = k)
                
                

            if 'cardinality' in multi_table_metrics:
                for score in cardinality[table_name]:
                    metrics_report = add_metric(metrics_report, 'cardinality', score, table_name=table_name, k = k)

    for table_name in metrics_report["metrics"]["single_table"].keys():
        for metric, metrics in metrics_report["metrics"]["single_table"][table_name].items():
            scores = metrics["scores"]
            metrics["mean"] = np.mean(scores)
            metrics["std"] = np.std(scores)
            metrics["min"] = np.min(scores)
            metrics["max"] = np.max(scores)
            metrics["median"] = np.median(scores)

    for metric, metrics in metrics_report["metrics"]["multi_table"].items():
        scores = metrics["scores"]
        metrics["mean"] = np.mean(scores)
        metrics["std"] = np.std(scores)
        metrics["min"] = np.min(scores)
        metrics["max"] = np.max(scores)
        metrics["median"] = np.median(scores)
            
    if save_report:
        with open(f"metrics_report/{dataset_name}_{method_name}.json", "w") as f:
            json.dump(metrics_report, f, indent=4)
    
    return metrics_report
