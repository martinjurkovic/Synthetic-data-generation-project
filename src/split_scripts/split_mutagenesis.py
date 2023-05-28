# %%
import rike.split_utils as utils
# %%
# READ DATA
dataset_name = "mutagenesis"
original_data = utils.read_tables(dataset_name)

# %%
# ADD PRIMARY KEY TO DEPENDENT TABLES THAT DON'T HAVE ONE
original_data['bond'] = utils.add_primary_key(original_data['bond'], 'bond_id')

# %%
# SPLIT DATASET 10 FOLD

molecule_folds = utils.split_k_fold(original_data["molecule"])

# %%
atom_folds = utils.split_k_fold_on_parent(
    molecule_folds, original_data["atom"], [("molecule_id", "molecule_id")])
# len(atom_folds[0][0])

# %%
bond_folds = utils.split_k_fold_on_parent(atom_folds, original_data["bond"],
                                          [("atom1_id", "atom_id"), ("atom2_id", "atom_id")])
# len(bond_folds[0][1])

# %%
utils.save_folds(
    [molecule_folds, atom_folds, bond_folds],
    dataset_name,
    ["molecule", "atom", "bond"])
# %%
