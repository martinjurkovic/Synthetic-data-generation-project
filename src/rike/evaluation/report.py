import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from rike.generation import sdv_metadata
from rike.utils import get_train_test_split, read_original_tables
from rike.evaluation.metrics import ks_test, cardinality_similarity


def generate_report(dataset_name, method_name, single_table_metrics=[ks_test], multi_table_metrics = ['cardinality'], save_report=False):
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
    # categorical = {}
    # for table, fields in metadata.to_dict()['tables'].items():
    #     categorical[table] = {}
    #     for field, values in fields['fields'].items():
    #         if (values['type'] == 'id' and values['subtype'] == 'string') or values['type'] == 'categorical':
    #             categorical[table][field] = tables_original[table][field].unique()
    #             print(table, categorical[table][field])

    for k in tqdm(range(10)):
        tables_orig_train, tables_orig_test = get_train_test_split(
            dataset_name, test_fold_index=k, synthetic=False)
        tables_synthetic_train, tables_synthetic_test = get_train_test_split(
            dataset_name, test_fold_index=k, synthetic=True, method_name=method_name)
        
        # for table, fields in categorical.items():
        #     for field, values in fields.items():
        #         # remove nan values from values numpy array
        #         values = values[~pd.isnull(values)]
        #         tables_orig_test[table][field] = pd.Categorical(tables_orig_test[table][field], categories=values)
        #         tables_synthetic_test[table][field] = pd.Categorical(tables_synthetic_test[table][field], categories=values)
        
        
        # SDV cardinality shape similarity
        if 'cardinality' in multi_table_metrics:
            cardinality = cardinality_similarity(tables_orig_test,
                            tables_synthetic_test,
                            metadata)
            


        # Single table metrics
        for table_name in tables_orig_test.keys():
            for metric in single_table_metrics:
                metric_name = metric.__name__
                metric_value = metric(tables_orig_test[table_name], tables_synthetic_test[table_name],
                                      original_train=tables_orig_train[table_name], synthetic_train=tables_synthetic_train[table_name], metadata = metadata.to_dict()['tables'][table_name])
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
                
        # Multi table metrics

    for table_name in metrics_report["metrics"]["single_table"].keys():
        for metric in metrics_report["metrics"]["single_table"][table_name].keys():
            metrics = metrics_report["metrics"]["single_table"][table_name][metric]
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
