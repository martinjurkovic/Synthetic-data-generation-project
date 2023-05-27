# %%
import json
import warnings
import argparse

from rike.evaluation.metrics import (logistic_detection, random_forest_detection,
                                     svm_detection, knn_detection, mlp_detection, xgboost_detection,
                                     parent_child_logistic_detection, parent_child_xgb_detection)
from rike.evaluation.report import generate_report
from rike.utils import get_highest_fold



args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="zurich")
args.add_argument("--method", type=str, default="sdv")
args.add_argument("--limit", type=int, default=-1)
args, unknown = args.parse_known_args()

limit = args.limit
if limit == -1:
    limit = get_highest_fold(args.dataset_name, args.method, evaluation=True) + 1
elif args.method == "subsample":
    if limit < 4:
      warnings.warn("Subsample method requires at least 4 folds. Setting limit to 4.")
      limit = 4
    elif limit % 2 != 0:
        warnings.warn("Subsample method requires an even number of folds. Setting limit to limit + 1.")
        limit += 1


# %%
single_table_metrics = [
                        #logistic_detection,
                        #random_forest_detection,
                        #svm_detection,
                        #knn_detection,
                        xgboost_detection,
                        #mlp_detection,
                     
                        ]

multi_table_metrics = [
                       # 'cardinality',
                       # parent_child_xgb_detection,
                        #parent_child_logistic_detection,
                      ]


# %%
report = generate_report(args.dataset_name, args.method,
                         single_table_metrics=single_table_metrics, 
                         multi_table_metrics=multi_table_metrics,
                         statistical_results=True,
                         save_report=True,
                         limit=limit)
# print formatted report dict
print(json.dumps(report, indent=4, sort_keys=True))

# %%
