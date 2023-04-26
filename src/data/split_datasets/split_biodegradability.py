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
# CREATE METADATA
if False:
    metadata = Metadata()
    metadata.add_table(
        name="molecule",
        data=original_data["molecule"],
        primary_key="molecule_id",
    )

    metadata.add_table(
        name="atom",
        data=original_data["atom"],
        primary_key="atom_id",
    )

    metadata.add_table(
        name="bond",
        data=original_data["bond"],
    )

    metadata.add_table(
        name="gmember",
        data=original_data["gmember"],
    )

    metadata.add_table(
        name="group",
        data=original_data["group"],
        primary_key="group_id",
    )

    metadata.add_relationship(
        parent="molecule",
        child="atom",
        foreign_key="molecule_id",
    )

    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom_id",
    )
    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom_id2",
    )

    metadata.add_relationship(
        parent="atom",
        child="gmember",
        foreign_key="atom_id",
    )

    metadata.add_relationship(
        parent="group",
        child="gmember",
        foreign_key="group_id",
    )

    metadata.visualize()

    metadata_name = utils.save_metadata(metadata, dataset_name)
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
