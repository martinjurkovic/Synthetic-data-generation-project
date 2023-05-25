# %%
import argparse
import os
import json

from rike.utils import get_highest_fold, get_train_test_split, read_original_tables, conditionally_sample, add_number_of_children
from rike.generation import sdv_metadata
from rike.evaluation.metrics import get_scores_dict
from rike.evaluation.report import aggregate_results

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, log_loss
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sdmetrics.utils import HyperTransformer
import xgboost as xgb

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str,
                  default="telstra-competition-dataset")
args.add_argument("--method", type=str, default="REalTabFormer")
args.add_argument("--limit", type=int, default=2)
args.add_argument("--save-report", type=bool, default=True)
args, unknown = args.parse_known_args()

dataset_name = args.dataset_name
method_name = args.method

limit = args.limit
if limit == -1:
    limit = get_highest_fold(
        args.dataset_name, args.method, evaluation=True) + 1


def mergefiles(dfs):
    countfiles = len(dfs)

    for i in range(countfiles):
        if i == 0:
            dfm = dfs[i]
        else:
            dfm = pd.merge(dfm, dfs[i], on="id")

    return dfm


# %%
json_path = f"metrics_report/ML_efficacy.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        metrics_report = json.load(f)
else:
    metrics_report = {}

if dataset_name not in metrics_report:
    metrics_report[dataset_name] = {}
if method_name not in metrics_report[dataset_name]:
    metrics_report[dataset_name][method_name] = {}
if "orig" not in metrics_report[dataset_name][method_name]:
    metrics_report[dataset_name][method_name]["scores_orig"] = {}
if "synthetic" not in metrics_report[dataset_name][method_name]:
    metrics_report[dataset_name][method_name]["scores_synthetic"] = {}
if "scores" not in metrics_report[dataset_name][method_name]["scores_orig"]:
    metrics_report[dataset_name][method_name]["scores_orig"]["scores"] = []
if "scores" not in metrics_report[dataset_name][method_name]["scores_synthetic"]:
    metrics_report[dataset_name][method_name]["scores_synthetic"]["scores"] = []

scores_orig = metrics_report[dataset_name][method_name]["scores_orig"]["scores"]
scores_synth = metrics_report[dataset_name][method_name]["scores_synthetic"]["scores"]
for k in tqdm(range(min(10, limit))):
    tables_original = read_original_tables(dataset_name)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_original)
    tables_orig_train, tables_orig_test = get_train_test_split(
        dataset_name, leave_out_fold_num=k, synthetic=False,
        method_name=method_name, limit=limit, metadata=metadata)
    tables_synthetic_train, tables_synthetic_test = get_train_test_split(
        dataset_name, leave_out_fold_num=k, synthetic=True,
        method_name=method_name, limit=limit, metadata=metadata)

    # data cleaning
    table_sets = (tables_orig_train, tables_orig_test,
                  tables_synthetic_train, tables_synthetic_test)
    ht = HyperTransformer()
    for idx, table_set in enumerate(table_sets):
        table_set['train'].location = table_set['train'].location.str.lstrip(
            'location').astype(int)

        table_set['event_type'].event_type = table_set['event_type'].event_type.str.lstrip(
            '"event_type ')
        # table_set['event_type'] = pd.get_dummies(table_set['event_type'], columns=['event_type'])

        table_set['log_feature'].log_feature = table_set['log_feature'].log_feature.map(
            lambda x: x.lstrip('feature '))
        table_set['resource_type'].resource_type = table_set['resource_type'].resource_type.str.lstrip(
            'resource_type ')
        # table_set['resource_type'] = pd.get_dummies(table_set['resource_type'], columns=['resource_type'])

        table_set['severity_type'].severity_type = table_set['severity_type'].severity_type.str.lstrip(
            'severity_type ')
        # table_set['severity_type'] = pd.get_dummies(table_set['severity_type'], columns=['severity_type'])

        dfs = [table_set['train'], table_set['event_type'], table_set['log_feature'],
               table_set['resource_type'], table_set['severity_type']]
        result = mergefiles(dfs)
        # from result drop id, event_type_id, log_feature_id, resource_type_id, severity_type_id
        result = result.drop(['id', 'event_type_id', 'log_feature_id',
                             'resource_type_id', 'severity_type_id'], axis=1)

        if idx == 0:  # if orig train
            result = ht.fit_transform(result)
        else:
            result = ht.transform(result)

        table_set['clf_data'] = result

    # Train a classifier

    train_orig = tables_orig_train['clf_data']
    train_orig_X = train_orig.drop(['fault_severity'], axis=1)
    train_orig_y = train_orig['fault_severity']

    train_synth = tables_synthetic_train['clf_data']
    train_synth_X = train_synth.drop(['fault_severity'], axis=1)
    train_synth_y = train_synth['fault_severity']

    test = tables_orig_test['clf_data']
    test_X = test.drop(['fault_severity'], axis=1)
    test_y = test['fault_severity']

    clf = xgb.XGBClassifier()
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    # Train on original data
    print("Training on original data fold {}".format(k))
    model.fit(train_orig_X, train_orig_y)
    probs_orig = model.predict_proba(test_X)
    y_pred_orig = probs_orig.argmax(axis=1)
    # print classification report
    print("Original data fold {}".format(k))
    print(classification_report(test_y, y_pred_orig))
    class_report_dict = classification_report(
        test_y, y_pred_orig, output_dict=True)
    scores_orig.append(get_scores_dict(test_y, y_pred_orig, probs_orig))

    # Train on synthetic data
    print("Training on synthetic data fold {}".format(k))
    model.fit(train_synth_X, train_synth_y)
    probs_synthetic = model.predict_proba(test_X)
    y_pred_synthetic = probs_synthetic.argmax(axis=1)
    # print classification report
    print("Synthetic data fold {}".format(k))
    print(classification_report(test_y, y_pred_synthetic))
    scores_synth.append(get_scores_dict(
        test_y, y_pred_synthetic, probs_synthetic))

metrics_report[dataset_name][method_name]["scores_orig"] = aggregate_results(metrics_report[dataset_name][method_name]["scores_orig"])
metrics_report[dataset_name][method_name]["scores_synthetic"] = aggregate_results(metrics_report[dataset_name][method_name]["scores_synthetic"])

if args.save_report:
    with open(json_path, "w") as f:
        json.dump(metrics_report, f, indent=4, sort_keys=True)

# %%
