import os
import argparse

import tqdm
import pandas as pd
from rike import utils
from rike.generation import sdv_metadata
from realtabformer import REaLTabFormer


# # OLD 
# parent_df = pd.read_csv("/content/drive/MyDrive/MAG-1/DS-Project/colab/Rossman_data/source/store.csv")
# child_df = pd.read_csv("/content/drive/MyDrive/MAG-1/DS-Project/colab/Rossman_data/source/sales.csv")

# join_on = "Store"
# assert ((join_on in parent_df.columns) and
#         (join_on in child_df.columns))

# parent_model = REaLTabFormer(model_type="tabular")
# parent_model_path = 'models/realtabformer_rossmann_store_sales' # TODO: fix this path

# parent_model = REaLTabFormer(
#     model_type="tabular",
#     parent_realtabformer_path=parent_model_path,
# )
# parent_model = parent_model.load_from_dir(parent_model_path)
# parent_model.fit(parent_df.drop(join_on, axis=1))
# parent_samples = parent_model.sample(len(parent_df))

# parent_samples.to_csv(f'data/synthetic/rossmann-store-sales/REaLTabFormer/rossmann_store_fold_0.csv', index=False) # TODO: fix this path

# child_model = REaLTabFormer(
#     model_type="relational",
#     parent_realtabformer_path=parent_model_path,
#     output_max_length=None,
#     train_size=0.95,
#     batch_size=1)

# child_model.fit(
#     df=child_df.sample(n=10000, random_state=42),
#     df = child_df,
#     in_df=parent_df,
#     join_on=join_on)

dataset_name = 'rossmann-store-sales'
root_table_name = 'store'

# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = utils.get_train_test_split(dataset_name, k)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    # select parent and child tables
    parent_df = tables_train[root_table_name]
    child_df = tables_train['sales']
    # check if join key is in both tables
    join_on = metadata.to_dict()['tables']['sales']['foreign_key']
    assert ((join_on in parent_df.columns) and
            (join_on in child_df.columns))

    # init parent model
    parent_model = REaLTabFormer(model_type="tabular")
    parent_model_path = f'models/realtabformer/rossmann_store_sales/checkpoint_{root_table_name}_{k}' 
    # fit and save parent model
    parent_model.fit(parent_df.drop(join_on, axis=1))
    parent_model.save(parent_model_path)
    # load trained parent model
    parent_model = REaLTabFormer(
        model_type="tabular",
        parent_realtabformer_path=parent_model_path,
    )
    parent_model = parent_model.load_from_dir(parent_model_path)
    # sample from parent model
    parent_samples = parent_model.sample(tables_test[root_table_name].shape[0])

    # init child model
    child_model = REaLTabFormer(
        model_type="relational",
        parent_realtabformer_path=parent_model_path,
        output_max_length=None,
        train_size=0.95,
        batch_size=1)
    # fit child model
    child_model.fit(
        df = child_df,
        in_df=parent_df,
        join_on=join_on)
    # sample from child model
    child_samples = child_model.sample(
        input_unique_ids=parent_samples[join_on],
        input_df=parent_samples.drop(join_on, axis=1),
        gen_batch=64)
    # save child model
    child_model_path = f'models/realtabformer/rossmann_store_sales/checkpoint_sales_{k}'
    child_model.save(child_model_path)

    # save synthetic data
    synthetic_data = {
        root_table_name: parent_samples,
        'sales': child_samples
    }
    utils.save_data(dataset_name, synthetic_data, k, method='REaLTabFormer')
