# %%
import json
import argparse

from rike.evaluation.metrics import (ks_test, chisquare_test, mean_max_discrepency, 
                                     js_divergence, logistic_detection, random_forest_detection,
                                     svm_detection, knn_detection, mlp_detection, xgboost_detection,
                                     parent_child_logistic_detection, parent_child_xgb_detection)
from rike.evaluation.report import generate_report
from rike.utils import get_highest_fold



args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann-store-sales")
args.add_argument("--method", type=str, default="mostlyai")
args.add_argument("--limit", type=int, default=-1)
args, unknown = args.parse_known_args()

limit = args.limit
if limit == -1:
    limit = get_highest_fold(args.dataset_name, args.method) + 1


# %%
single_table_metrics = [
                        #ks_test,
                        logistic_detection,
                        random_forest_detection,
                        svm_detection,
                        knn_detection,
                        xgboost_detection,
                        mlp_detection,
                        # chisquare_test,
                        # mean_max_discrepency,
                        # js_divergence,
                        ]

multi_table_metrics = [
                        # 'cardinality',
                        parent_child_xgb_detection,
                        parent_child_logistic_detection,
                      ]

# %%
report = generate_report(args.dataset_name, args.method,
                         single_table_metrics=single_table_metrics, 
                         multi_table_metrics=multi_table_metrics,
                         save_report=True,
                         limit=limit)
# print formatted report dict
print(json.dumps(report, indent=4, sort_keys=True))

# %%
