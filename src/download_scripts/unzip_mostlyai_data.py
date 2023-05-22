import os
import zipfile
import argparse

import pandas as pd

from rike.utils import read_original_tables
# add args
args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann-store-sales")
args = args.parse_args()
dataset_name = args.dataset_name

base_path = 'data/downloads/mostlyai'
zip_fies = os.listdir(base_path)
for file in zip_fies:
    if file.endswith(".zip"):
        dataset = file.split('_')[0].lower()
        if dataset not in dataset_name.lower():
              continue
        tables_original = read_original_tables(dataset_name)
        unzip_path = os.path.join(base_path, dataset)
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(f'{base_path}/{file}', 'r') as zip_ref:
                    zip_ref.extractall(f'{unzip_path}')
        subfolders = os.listdir(unzip_path)
        for subfolder in subfolders:
               files = os.listdir(os.path.join(unzip_path, subfolder))
               for file in files:
                    table = file.split('_')[0].lower()
                    file_path = os.path.join(unzip_path, subfolder, file)
                    new_path = os.path.join('data/synthetic', dataset, 'mostlyai', f'{dataset}_{file}')
                    # create new folder if it does not exist
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    df = pd.read_csv(file_path)
                    orig_columns = tables_original[table].columns
                    df = df[[colname.lower() for colname in orig_columns]]
                    df.columns = orig_columns
                    if 'bond' in file:
                        df['type'][df['type'] == '_RARE_'] = -1
                        print(df.type.unique())
                    df.to_csv(new_path, index=False)
                     
        
