# %%
from rike.generation import sdv_utils
from rike.generation import sdv_metadata
from sdv.relational import HMA1

DATASET_NAME = "biodegradability"

# %%
# GENERATE SYNTHETIC DATA
for k in range(10):
    tables_train, tables_test = sdv_utils.get_train_test_split(DATASET_NAME, k)
    metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
    # Create HMA1 model
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    synthetic_data = model.sample(num_rows=tables_test["molecule"].shape[0])
    sdv_utils.save_SDV_data(synthetic_data, DATASET_NAME, k)


# %%
