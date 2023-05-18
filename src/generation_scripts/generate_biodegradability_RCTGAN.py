# %%
import pickle

import tqdm
import pandas as pd
from rike.generation import sdv_utils
from rike.generation import sdv_metadata
from rctgan import RCTGAN

# %%

DATASET_NAME = "biodegradability"

# GENERATE SYNTHETIC DATA
for k in tqdm.tqdm(range(10)):
    tables_train, tables_test = sdv_utils.get_train_test_split(DATASET_NAME, k)
    metadata = sdv_metadata.generate_metadata(DATASET_NAME, tables_train)
    # Create HMA1 model
    model = RCTGAN(metadata)
    # # ignores warnings being raised inside the RCTGAN package
    # with pd.option_context('mode.chained_assignment', None):
    model.fit(tables_train)
    pickle.dump(model, open(f'models/model_rctgan{DATASET_NAME}_fold_{k}.pickle', "wb" ) )
    synthetic_data = model.sample()
    sdv_utils.save_data(synthetic_data, DATASET_NAME, k, method='RCTGAN')
# %%