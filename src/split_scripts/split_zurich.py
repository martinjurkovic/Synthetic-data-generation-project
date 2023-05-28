# %%
import rike.split_utils as utils
# %%
# READ DATA
dataset_name = "zurich"
original_data = utils.read_tables(dataset_name)
# %%
## clean data
import pandas as pd
# drop nan values
# for table in original_data:
#     print(original_data[table].info())

# original_data['customers'] = original_data['customers'].dropna()
# original_data['claims'] = original_data['claims'].dropna()
# original_data['policies'] = original_data['policies'].dropna()

# convert the datetime columns to datetime type
original_data['customers']['date_of_birth'] = pd.to_datetime(original_data['customers']['date_of_birth'])
original_data['claims']['date_closed'] = pd.to_datetime(original_data['claims']['date_closed'])
original_data['claims']['event_date'] = pd.to_datetime(original_data['claims']['event_date'])
original_data['claims']['date_open'] = pd.to_datetime(original_data['claims']['date_open'])
original_data['policies']['underwriting_date'] = pd.to_datetime(original_data['policies']['underwriting_date'])
original_data['policies']['cancellation_or_end_date'] = pd.to_datetime(original_data['policies']['cancellation_or_end_date'])
original_data['policies']['first_end_date'] = pd.to_datetime(original_data['policies']['first_end_date'])

original_data['claims']['customer_id'] = original_data['claims']['customer_id'].astype('int', errors='ignore')
original_data['claims']['claim_id'] = original_data['claims']['claim_id'].astype('int', errors='ignore')
original_data['claims']['policy_id'] = original_data['claims']['policy_id'].astype('int', errors='ignore')
original_data['policies']['customer_id'] = original_data['policies']['customer_id'].astype('int', errors='ignore')

# convert int64 columns to int32
# for column in original_data['customers'].columns:
#     if original_data['customers'][column].dtype == 'int64':
#         original_data['customers'][column] = original_data['customers'][column].astype('int32')
# for column in original_data['claims'].columns:
#     if original_data['claims'][column].dtype == 'int64':
#         original_data['claims'][column] = original_data['claims'][column].astype('int32')
# for column in original_data['policies'].columns:
#     if original_data['policies'][column].dtype == 'int64':
#         original_data['policies'][column] = original_data['policies'][column].astype('int32')

# original_data['policies']['customer_id'] = original_data['policies']['customer_id'].astype('int32')

# original_data['customers'] = original_data['customers'][original_data['customers']['customer_id'].isin(original_data['policies']['customer_id'])]
# original_data['claims'] = original_data['claims'][original_data['claims']['policy_id'].isin(original_data['policies']['policy_id'])]

# for table in original_data:
#     print(original_data[table].info())
# %%
frac = 100000 / len(original_data["customers"])
original_data["customers"] = original_data["customers"].sample(frac=0.001, random_state=42)

# %%
# %%
# SPLIT DATASET 10 FOLD

customer_folds = utils.split_k_fold(original_data["customers"])

# %%
policy_folds = utils.split_k_fold_on_parent(
    customer_folds, original_data["policies"], 
    [("customer_id", "customer_id")])

# %%
claim_folds = utils.split_k_fold_on_parent(
    policy_folds, original_data["claims"],
    [("policy_id", "policy_id")])


# %%
utils.save_folds(
    [customer_folds, policy_folds, claim_folds],
    dataset_name,
    ["customers", "policies", "claims"],)
# %%
