import os
import zipfile

import pandas as pd

base_path = 'data/downloads/mostlyai'
zip_fies = os.listdir(base_path)
for file in zip_fies:
    if file.endswith(".zip"):
        dataset = file.split('_fold')[0]
        unzip_path = os.path.join(base_path, dataset)
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(f'{base_path}/{file}', 'r') as zip_ref:
                    zip_ref.extractall(f'{unzip_path}')
        subfolders = os.listdir(unzip_path)
        for subfolder in subfolders:
               files = os.listdir(os.path.join(unzip_path, subfolder))
               for file in files:
                    file_path = os.path.join(unzip_path, subfolder, file)
                    new_path = os.path.join('data/synthetic', dataset, 'mostlyai', f'{dataset}_{file}')
                    df = pd.read_csv(file_path)
                    print(file_path)
                    if 'bond' in file:
                        df['type'][df['type'] == '_RARE_'] = -1
                        print(df.type.unique())
                    df.to_csv(new_path, index=False)
                     
        
