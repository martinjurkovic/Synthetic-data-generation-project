# %%
import os
import argparse
import pickle

import tqdm
import pandas as pd
from rike import utils
from rike.generation import sdv_metadata
from rike import logging_config
from rctgan import RCTGAN
import torch

# %%
args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="biodegradability")
args.add_argument("--start_fold", type=int, default=0)
args = args.parse_args()

dataset_name = args.dataset_name
start_fold = args.start_fold
root_table_name = sdv_metadata.get_root_table(dataset_name)

logger = logging_config.logger
logger.error(f"CUDA: {torch.cuda.is_available()}")

logger.error("START...")

CWD_PROJECT = os.getcwd().split(
    'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
base_path = CWD_PROJECT + '/data/synthetic/' + dataset_name + '/RCTGAN/'

# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(start_fold, 10)):
    logger.error(f"STARTIG FOLD {k}!")
    model_save_path = f'models/rctgan/{dataset_name}/model_{k}.pickle'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    tables_train, tables_test = utils.get_train_test_split(dataset_name, k)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)

    assert root_table_name in metadata.to_dict()['tables'].keys(), \
        f'Root table {root_table_name} not in metadata tables'
    # check if data already exists
    generated = False
    for table_name in metadata.to_dict()['tables'].keys():
        path = base_path + f'{dataset_name}_{table_name}_fold_{k}.csv'
        if os.path.exists(path):
            generated = True
            break
    if generated:
        logger.error("Loading model from pickled file...")
        model = pickle.load(open(model_save_path, "rb" ) )
        logger.error("Loaded successfully!")
    else:
        logger.error(f"Initializing RCTGAN model for fold: {k}...")
        model = RCTGAN(metadata)
        # ignores warnings being raised inside the RCTGAN package
        with pd.option_context('mode.chained_assignment', None):
            logger.error(f"FITTING RCTGAN model for fold: {k}...")
            model.fit(tables_train)
            logger.error("DONE FITTING!")
        pickle.dump(model, open(model_save_path, "wb" ) )
    logger.error("SAMPLING AND SAVING SYNTHETIC DATA...")
    synthetic_data = model.sample(num_rows=tables_test[root_table_name].shape[0])
    utils.save_data(synthetic_data, dataset_name, k, method='RCTGAN')
# %%