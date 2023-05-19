# %%
import json
import argparse

from rike.evaluation.metrics import (ks_test, chisquare_test, mean_max_discrepency, 
                                     js_divergence, logistic_detection, random_forest_detection,
                                     svm_detection, knn_detection, mlp_detection, xgboost_detection)
from rike.evaluation.report import generate_report



args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="biodegradability")
args.add_argument("--method-name", type=str, default="rctgan")
args, unknown = args.parse_known_args()


# %%
single_table_metrics = [
                        #ks_test,
                        #logistic_detection,
                        #random_forest_detection,
                        #svm_detection,
                        #knn_detection,
                        xgboost_detection,
                        #mlp_detection,
                        # chisquare_test,
                        # mean_max_discrepency,
                        # js_divergence,
                        ]

# %%
report = generate_report(args.dataset_name, args.method_name,
                         single_table_metrics=single_table_metrics, save_report=True)
# print formatted report dict
print(json.dumps(report, indent=4))

# %%