# %%
from rike.generation import sdv_utils
from rike.generation import sdv_metadata
from sdv.relational import HMA1
import tqdm

DATASET_NAME = "biodegradability"

# %%
# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = sdv_utils.get_train_test_split(DATASET_NAME, k)
    sdv_utils.save_train_test_split(DATASET_NAME, k, tables_train, tables_test)
    # metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
    

# %%
# tables_train, tables_test = sdv_utils.get_train_test_split(DATASET_NAME, 0)
# metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
# metadata.visualize()
# %%
