import os
import argparse

import tqdm
import pandas as pd
from rike import utils
from rike.generation import sdv_metadata
from realtabformer import REaLTabFormer
from rike import logging_config
import glob
import wandb

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann-store-sales")
args.add_argument("--full_sensitivity", type=bool, default=True)
args = args.parse_args()
dataset_name = args.dataset_name
full_sensitivity = args.full_sensitivity
root_table_name = sdv_metadata.get_root_table(dataset_name)

os.environ["WANDB_PROJECT"]=f"REALTABFORMER_{dataset_name}"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

logger = logging_config.logger

logger.error("START...")
logger.info("START INFO...")

batch_size_parent = 1024
batch_size_child = 10
retrain = True

# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    logger.error("Generating synthetic data for %s, fold %d", dataset_name, k)
    synthetic_data = {}
    
    tables_train, tables_test = utils.get_train_test_split(dataset_name, k)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    
    # GENERATE PARENT TABLE
    parent_df = tables_train[root_table_name]
    logger.error(parent_df.head())
    # check if join key is in both tables
    join_on = metadata.to_dict()['tables'][root_table_name]['primary_key']

    parent_model_path = f'models/realtabformer/{dataset_name}/checkpoint_{root_table_name}_{k}' 
    # init parent model
    if retrain or not(os.path.exists(parent_model_path) and len(os.listdir(parent_model_path)) > 0):
        parent_model = REaLTabFormer(model_type="tabular", batch_size=batch_size_parent, report_to="wandb")
        os.makedirs(parent_model_path, exist_ok=True)
        # fit and save parent model
        parent_model.fit(parent_df.drop(join_on, axis=1), full_sensitivity=full_sensitivity)
        parent_model.save(parent_model_path)
    # load trained parent model
    directories = list(filter(os.path.isdir, glob.glob(f"{parent_model_path}/id*")))
    directories.sort(key=lambda x: os.path.getmtime(x))
    parent_model_path = directories[-1]
    parent_model = REaLTabFormer(
        model_type="tabular",
        parent_realtabformer_path=parent_model_path,
    )
    parent_model = parent_model.load_from_dir(parent_model_path)
    # sample from parent model
    parent_samples = parent_model.sample(tables_test[root_table_name].shape[0])
    parent_samples.index.name = join_on
    parent_samples = parent_samples.reset_index()
    synthetic_data[root_table_name] = parent_samples

    #GENERATE CHILD TABLES
    for child_table_name in metadata.get_children(root_table_name):
        child_model_path = f'models/realtabformer/{dataset_name}/checkpoint_{child_table_name}_{k}'
        child_df = tables_train[child_table_name]
        logger.error(child_df.head())
        assert ((join_on in parent_df.columns) and
                (join_on in child_df.columns))
        # init child model
        if retrain or not(os.path.exists(child_model_path) and len(os.listdir(child_model_path)) > 0):
            child_model = REaLTabFormer(
                model_type="relational",
                parent_realtabformer_path=parent_model_path,
                output_max_length=None,
                train_size=0.95,
                batch_size=batch_size_child,
                report_to="wandb")
            # fit child model
            child_model.fit(
                df = child_df,
                in_df=parent_df,
                join_on=join_on)
            # save child model
            os.makedirs(child_model_path, exist_ok=True)
            child_model.save(child_model_path)

        directories = list(filter(os.path.isdir, glob.glob(f"{child_model_path}/id*")))
        directories.sort(key=lambda x: os.path.getmtime(x))
        child_model_path = directories[-1]
        child_model = REaLTabFormer(
            model_type="relational",
                parent_realtabformer_path=parent_model_path,
                output_max_length=None,
                train_size=0.95,
                batch_size=batch_size_child,
                report_to="wandb"
        )
        child_model = child_model.load_from_dir(child_model_path)
        
        child_samples = child_model.sample(
            input_unique_ids=parent_samples[join_on],
            input_df=parent_samples.drop(join_on, axis=1),
            gen_batch=64)
        
        child_samples.index.name = join_on
        child_samples = child_samples.reset_index()
        
        synthetic_data[child_table_name] = child_samples
        

    utils.save_data(dataset_name=dataset_name, tables_synthetic=synthetic_data, leave_out_fold_num=k, method='REaLTabFormer', index=False)
