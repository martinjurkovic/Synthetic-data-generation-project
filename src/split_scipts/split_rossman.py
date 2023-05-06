# %%
import rike.split_utils as utils
import pandas as pd

# %%
DATASET_NAME = "rossmann-store-sales"
original_data = utils.read_tables(DATASET_NAME)

# %%
# Join train and test data
# original_data["sales"] = pd.concat([original_data["train"], original_data["test"]])
# %%
# SPLIT DATASET 10 FOLD
store_folds = utils.split_k_fold(original_data["store"])

# %%
sales_folds = utils.split_k_fold_on_parent(
    store_folds, original_data["test"], [("Store", "Store")])

# %%
utils.save_folds(
    [store_folds, sales_folds],
    DATASET_NAME,
    ["store", "test"],
)
# %%
