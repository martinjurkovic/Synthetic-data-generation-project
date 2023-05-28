# %%
import rike.split_utils as utils
import pandas as pd

# %%
DATASET_NAME = "telstra-competition-dataset"
original_data = utils.read_tables(DATASET_NAME)

# %%
original_data['severity_type'] = utils.add_primary_key(original_data['severity_type'], 'severity_type_id')
original_data['resource_type'] = utils.add_primary_key(original_data['resource_type'], 'resource_type_id')
original_data['log_feature'] = utils.add_primary_key(original_data['log_feature'], 'log_feature_id')
original_data['event_type'] = utils.add_primary_key(original_data['event_type'], 'event_type_id')

# %%
# SPLIT DATASET 10 FOLD
train_folds = utils.split_k_fold(original_data["train"])
# %%
severity_folds = utils.split_k_fold_on_parent(
    train_folds, original_data["severity_type"], [("id", "id")])

resource_folds = utils.split_k_fold_on_parent(
    train_folds, original_data["resource_type"], [("id", "id")])

log_folds = utils.split_k_fold_on_parent(
    train_folds, original_data["log_feature"], [("id", "id")])

event_folds = utils.split_k_fold_on_parent(
    train_folds, original_data["event_type"], [("id", "id")])

# %%
utils.save_folds(
    [train_folds, severity_folds, resource_folds, log_folds, event_folds],
    DATASET_NAME,
    ["train", "severity_type", "resource_type", "log_feature", "event_type"],
)
# %%
