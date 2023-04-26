# %%
import utils
import os
import pandas as pd
from sdv import Metadata
from sdv.utils import display_tables
from sklearn.model_selection import KFold


# %%
# READ DATA
dataset_name = "biodegradability"
original_data = utils.read_tables(dataset_name)

# %%
# SPLIT DATASET 10 FOLD

molecule_folds = utils.split_k_fold(original_data["molecule"])
group_folds = utils.split_k_fold(original_data["group"])

# %%
atom_folds = utils.split_k_fold_on_parent(
    molecule_folds, original_data["atom"], [("molecule_id", "molecule_id")])
# len(atom_folds[0][0])

# %%
bond_folds = utils.split_k_fold_on_parent(atom_folds, original_data["bond"],
                                          [("atom_id", "atom_id"), ("atom_id2", "atom_id")])
# len(bond_folds[0][1])
# %%
gmember_folds = utils.split_k_fold_on_multiple_parents(
    parents_folds=[atom_folds, group_folds],
    child_table=original_data["gmember"],
    split_col_names=[
        [("atom_id", "atom_id")],
        [("group_id", "group_id")],
    ]
)

# %%
utils.save_folds(
    [molecule_folds, atom_folds, bond_folds, gmember_folds, group_folds],
    dataset_name,
    ["molecule", "atom", "bond", "gmember", "group"])
# %%
