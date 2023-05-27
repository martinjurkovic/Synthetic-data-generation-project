import os
from rike.evaluation.metrics import (logistic_detection, random_forest_detection,
                                     svm_detection, knn_detection, mlp_detection, xgboost_detection,
                                     parent_child_logistic_detection, parent_child_xgb_detection)
from rike.evaluation.report import generate_report
from rike.utils import get_highest_fold
from rike import logging_config
from realtabformer import REaLTabFormer

logger = logging_config.logger
CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'

# biodegradability, telstra-competition-dataset, mutagenesis, rossmann-store-sales, zurich
datasets = ['biodegradability', 'telstra-competition-dataset', 'mutagenesis', 'rossmann-store-sales', 
            # 'zurich',
            ]
# gretel, mostlyai, RCTGAN, SDV, RealTabFormer, subsample
methods = ['gretel', 'mostlyai', 'RCTGAN', 'SDV', 'RealTabFormer', 'subsample']

single_table_metrics = [
                        logistic_detection,
                        #random_forest_detection,
                        #svm_detection,
                        #knn_detection,
                        xgboost_detection,
                        #mlp_detection,
                     
                        ]

multi_table_metrics = [
                       # 'cardinality',
                        parent_child_xgb_detection,
                        #parent_child_logistic_detection,
                      ]

for method in methods:
    for dataset in datasets:
        path = os.path.join(CWD_PROJECT, 'data', 'splits', dataset)            
        if not os.path.exists(path):
           logger.error(f"No splits found for dataset: {dataset} and method {method}! Skipping...")
           continue

        logger.error(f"STARTING {method} on {dataset}!")
        limit = get_highest_fold(dataset, method, evaluation=True) + 1

        try:
          report = generate_report(dataset, method,
                          single_table_metrics=single_table_metrics, 
                          multi_table_metrics=multi_table_metrics,
                          statistical_results=True,
                          save_report=True,
                          limit=limit)
        except Exception as e:
          logger.error(f"FAILED {method} on {dataset}!")
          logger.error(e)