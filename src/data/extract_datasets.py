import os
import shutil
import zipfile

DATA_DIR = 'data'

#for zip in datadir unzip the dataset
for file in os.listdir(DATA_DIR):
    if file.endswith('.zip'):
        zip_path = f'{DATA_DIR}/{file}'
        unzip_path = f'{DATA_DIR}/{file[:-4]}'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        # for each file in the unzipped dataset check if it is a zip file
        for file in os.listdir(unzip_path):
            if file.endswith('.zip'):
                with zipfile.ZipFile(f'{unzip_path}/{file}', 'r') as zip_ref:
                    zip_ref.extractall(f'{unzip_path}')
                # remove the zip file
                os.remove(f'{unzip_path}/{file}')
        # remove the zip file
        os.remove(zip_path)
        # delete the __MACOSX folder if it exists
        if os.path.exists(f'{unzip_path}/__MACOSX'):
            shutil.rmtree(f'{unzip_path}/__MACOSX')
            