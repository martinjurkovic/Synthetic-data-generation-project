# %%
import argparse
from rike.generation import generation_utils
from rike.generation import sdv_metadata
from sdv.relational import HMA1
import tqdm

args = argparse.ArgumentParser()
args.add_argument("--dataset_name", type=str, default="rossmann-store-sales")
args.add_argument("--root_table_name", type=str, default="store")
args = args.parse_args()

dataset_name = args.dataset_name
root_table_name = args.root_table_name

# %%
# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = generation_utils.get_train_test_split(dataset_name, k)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    # Create HMA1 model
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    synthetic_data = model.sample(num_rows=tables_test[root_table_name].shape[0])
    generation_utils.save_data(synthetic_data, dataset_name, k, method='SDV')

# %%
# tables_train, tables_test = generation_utils.get_train_test_split(dataset_name, 0)
# metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
# metadata.visualize()

# %%
