import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from rike.generation import sdv_metadata
from rike.utils import get_train_test_split, read_original_tables
from rike.evaluation.metrics import ks_test, cardinality_similarity


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
        
        
        # SDV cardinality shape similarity
        if 'cardinality' in multi_table_metrics:
            cardinality = cardinality_similarity(tables_orig_test,
                            tables_synthetic_test,
                            metadata)
            
        for table, fields in metadata.to_dict()['tables'].items():
            for field, values in fields['fields'].items():
                # convert the datetime columns to datetime type
                if values['type'] == 'datetime':
                    tables_orig_test[table][field] = pd.to_datetime(tables_orig_test[table][field], format=values['format'])
                    tables_synthetic_test[table][field] = pd.to_datetime(tables_synthetic_test[table][field], format=values['format'])
                    tables_orig_train[table][field] = pd.to_datetime(tables_orig_train[table][field], format=values['format'])
                    tables_synthetic_train[table][field] = pd.to_datetime(tables_synthetic_train[table][field], format=values['format'])
            # sort the columns of all tables
            column_order = tables_orig_test[table].columns
            tables_orig_train[table] = tables_orig_train[table].reindex(column_order, axis=1)
            tables_synthetic_test[table] = tables_synthetic_test[table].reindex(column_order, axis=1)
            tables_synthetic_train[table] = tables_synthetic_train[table].reindex(column_order, axis=1)

            # TODO: rework data stratification
            # This is probably not ok, we should stratify only the parent table 
            # and then sample the child tables accordingly
            if len(tables_synthetic_test[table]) > len(tables_orig_test[table]):
                tables_synthetic_test[table] = tables_synthetic_test[table].sample(n=len(tables_orig_test[table]))

            if len(tables_synthetic_train[table]) > len(tables_orig_train[table]):
                tables_synthetic_train[table] = tables_synthetic_train[table].sample(n=len(tables_orig_train[table]))

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
            if metric_name not in metrics_report["metrics"]["multi_table"].keys():
                metrics_report["metrics"]["multi_table"][metric_name] = {
                    "scores": []}
            metrics_report["metrics"]["multi_table"][metric_name]["scores"].append(metric_value)
            print(metric_name, metric_value)

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
            for metric in single_table_metrics:
                metric_name = metric.__name__
                metric_value = metric(tables_orig_test[table_name], tables_synthetic_test[table_name],
                                      original_train=tables_orig_train[table_name], 
                                      synthetic_train=tables_synthetic_train[table_name], 
                                      metadata = metadata.to_dict()['tables'][table_name],
                                      save_path=f"metrics_report/{dataset_name}/{method_name}/{metric_name}/probabilities/{table_name}_{k}.csv")
                if table_name not in metrics_report["metrics"]["single_table"].keys():
                    metrics_report["metrics"]["single_table"][table_name] = {}
                if metric_name not in metrics_report["metrics"]["single_table"][table_name] or k == 0:
                    metrics_report["metrics"]["single_table"][table_name][metric_name] = {
                        "scores": []}
                metrics_report["metrics"]["single_table"][table_name][metric_name]["scores"].append(
                    metric_value)
            if 'cardinality' in multi_table_metrics:
                if 'cardinality' not in metrics_report["metrics"]["single_table"][table_name] or k == 0:
                    metrics_report["metrics"]["single_table"][table_name]['cardinality'] = {
                        "scores": []}
                for score in cardinality[table_name]:
                    metrics_report["metrics"]["single_table"][table_name]['cardinality']["scores"].append(score)

    for table_name in metrics_report["metrics"]["single_table"].keys():
        for metric in metrics_report["metrics"]["single_table"][table_name].keys():
            metrics = metrics_report["metrics"]["single_table"][table_name][metric]
            scores = metrics["scores"]
            metrics["mean"] = np.mean(scores)
            metrics["std"] = np.std(scores)
            metrics["min"] = np.min(scores)
            metrics["max"] = np.max(scores)
            metrics["median"] = np.median(scores)

    for metric in metrics_report["metrics"]["multi_table"].keys():
        metrics = metrics_report["metrics"]["multi_table"][metric]
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
