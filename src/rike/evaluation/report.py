from rike.evaluation.metrics import ks_test, chisquare_test, mean_max_discrepency, js_divergence
from rike.utils import get_train_test_split
import json
import numpy as np


def generate_report(dataset_name, method_name, single_table_metrics=[ks_test], save_report=False):
    metrics_report = {
        "dataset_name": dataset_name,
        "metrics": {
            "single_table": {
            },
            "multi_table": {
            },
        }
    }
    for k in range(10):
        tables_orig_train, tables_orig_test = get_train_test_split(
            "biodegradability", test_fold_index=k, synthetic=False)
        tables_sdv_train, tables_sdv_test = get_train_test_split(
            "biodegradability", test_fold_index=k, synthetic=True, method_name=method_name)

        # Single table metrics
        for table_name in tables_orig_test.keys():
            for metric in single_table_metrics:
                metric_name = metric.__name__
                metric_value = metric(
                    tables_orig_test[table_name], tables_sdv_test[table_name])
                if table_name not in metrics_report["metrics"]["single_table"].keys():
                    metrics_report["metrics"]["single_table"][table_name] = {}
                if metric_name not in metrics_report["metrics"]["single_table"][table_name]:
                    metrics_report["metrics"]["single_table"][table_name][metric_name] = {
                        "scores": [], "mean": None, "std": None}
                metrics_report["metrics"]["single_table"][table_name][metric_name]["scores"].append(
                    metric_value)

        # Multi table metrics
        # TODO: Add sdv cardinality

    for table_name in metrics_report["metrics"]["single_table"].keys():
        for metric in metrics_report["metrics"]["single_table"][table_name].keys():
            metrics_report["metrics"]["single_table"][table_name][metric]["mean"] = np.mean(
                metrics_report["metrics"]["single_table"][table_name][metric]["scores"])
            metrics_report["metrics"]["single_table"][table_name][metric]["std"] = np.std(
                metrics_report["metrics"]["single_table"][table_name][metric]["scores"])
            
    if save_report:
        with open(f"metrics_report/{dataset_name}_{method_name}.json", "w") as f:
            json.dump(metrics_report, f, indent=4)
    
    return metrics_report
