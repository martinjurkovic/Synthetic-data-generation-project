# %%
from rike.generation import generation_utils
from rike.generation import sdv_metadata
from sdv.relational import HMA1
import tqdm

DATASET_NAME = "rossmann-store-sales"
root_table_name = "store"

# %%
# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = generation_utils.get_train_test_split(DATASET_NAME, k)
    metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
    # Create HMA1 model
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    synthetic_data = model.sample(num_rows=tables_test[root_table_name].shape[0])
    generation_utils.save_data(synthetic_data, DATASET_NAME, k, method='SDV')

# %%
# tables_train, tables_test = sdv_utils.get_train_test_split(DATASET_NAME, 0)
# metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
# metadata.visualize()

# %%
