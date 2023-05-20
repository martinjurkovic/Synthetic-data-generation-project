# %%
import rike.split_utils as utils
# %%
# READ DATA
dataset_name = "zurich"
original_data = utils.read_tables(dataset_name)

# %%
# %%
# SPLIT DATASET 10 FOLD

customer_folds = utils.split_k_fold(original_data["customers"])

# %%
policy_folds = utils.split_k_fold_on_parent(
    customer_folds, original_data["policies"], 
    [("customer_id", "customer_id")])

# %%
claim_folds = utils.split_k_fold_on_multiple_parents(
    parents_folds=[customer_folds, policy_folds],
    child_table=original_data["claims"],
    split_col_names=[
        [("customer_id", "customer_id")],
        [("policy_id", "policy_id")],
    ]
)

# %%
utils.save_folds(
    [customer_folds, policy_folds, claim_folds],
    dataset_name,
    ["customers", "policies", "claims"],)
# %%
