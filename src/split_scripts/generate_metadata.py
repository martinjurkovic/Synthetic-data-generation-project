from tqdm import tqdm

import rike.split_utils as utils
from rike.generation import sdv_metadata

datasets = ['biodegradability', 'rossmann-store-sales', 'coupon-purchase-prediction', 'telstra-competition-dataset', 'zurich', 'mutagenesis']
for dataset_name in tqdm(datasets):
    original_data = utils.read_tables(dataset_name)
    metadata = sdv_metadata.generate_metadata(dataset_name, original_data, save=True)