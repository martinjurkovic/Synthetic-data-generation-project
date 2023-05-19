# %%
import os
import argparse
import pickle

import tqdm
import pandas as pd
from rike.generation import generation_utils
from rike.generation import sdv_metadata
from rctgan import RCTGAN

# %%

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann-store-sales")
args = args.parse_args()

dataset_name = args.dataset_name

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
base_path = CWD_PROJECT + '/data/synthetic/' + dataset_name + '/RCTGAN/'

# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = generation_utils.get_train_test_split(dataset_name, k)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    
    # check if data already exists
    generated = False
    for table_name in metadata.to_dict()['tables'].keys():
        path = base_path + f'{dataset_name}_{table_name}_fold_{k}.csv'
        if os.path.exists(path):
            generated = True
            break
    if generated:
        continue

    model = RCTGAN(metadata)
    # ignores warnings being raised inside the RCTGAN package
    with pd.option_context('mode.chained_assignment', None):
        model.fit(tables_train)
    pickle.dump(model, open(f'models/model_rctgan{dataset_name}_fold_{k}.pickle', "wb" ) )
    synthetic_data = model.sample()
    generation_utils.save_data(synthetic_data, dataset_name, k, method='RCTGAN')
# %%