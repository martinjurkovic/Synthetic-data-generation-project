# %%
import os
import pickle
import argparse
from rike import utils
from rike.generation import sdv_metadata
from sdv.relational import HMA1
import tqdm
from rike import logging_config

logger = logging_config.logger

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann-store-sales")
args.add_argument("--limit", type=int, default=10)
args = args.parse_args()

dataset_name = args.dataset_name
limit = args.limit
root_table_name = sdv_metadata.get_root_table(dataset_name)

# %%
# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(limit)):
    model_save_path = f'models/sdv/{dataset_name}/model_{k}.pickle'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    logger.warning("Generating synthetic data for %s, fold %d", dataset_name, k)
    tables_train, tables_test = utils.get_train_test_split(dataset_name, k, limit=limit)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    # Create HMA1 model
    logger.warning("Fitting HMA1 model...")
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, open(model_save_path, "wb" ) )
    logger.warning("Done!")
    logger.warning("Sampling synthetic data...")
    synthetic_data = model.sample(num_rows=tables_test[root_table_name].shape[0])
    logger.warning("Done!")
    logger.warning("Saving synthetic data...")
    utils.save_data(synthetic_data, dataset_name, k, method='SDV')
    logger.warning(f"Done with fold {k}!\n\n")
logger.warning("Done with all folds!")

# %%
# tables_train, tables_test = utils.get_train_test_split(dataset_name, 0)
# metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
# metadata.visualize()

# %%
