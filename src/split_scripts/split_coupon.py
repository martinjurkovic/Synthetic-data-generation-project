# %%
import rike.split_utils as utils
# %%
# READ DATA
dataset_name = "coupon-purchase-prediction"
original_data = utils.read_tables(dataset_name)

# %%
# ADD PRIMARY KEY TO DEPENDENT TABLES THAT DON'T HAVE ONE
# original_data['coupon_area_train'] = utils.add_primary_key(
#     original_data['coupon_area_train'], 'coupon_area_train_id')
original_data['coupon_detail_train'] = utils.add_primary_key(
    original_data['coupon_detail_train'], 'coupon_detail_train_id')
original_data['coupon_visit_train'] = utils.add_primary_key(
    original_data['coupon_visit_train'], 'coupon_visit_train_id')

# %%
# SPLIT DATASET 10 FOLD
# sample 10% of users
users_sampled = original_data["user_list"].sample(frac=0.05, random_state=42)
user_folds = utils.split_k_fold(users_sampled)
# coupon_folds = utils.split_k_fold(original_data["coupon_list_train"])

# %%
# coupon_area_folds = utils.split_k_fold_on_parent(
#     coupon_folds, original_data["coupon_area_train"], [("COUPON_ID_hash", "COUPON_ID_hash")])

# %%
# coupon_detail_folds = utils.split_k_fold_on_multiple_parents(
#     parents_folds=[coupon_folds, user_folds],
#     child_table=original_data["coupon_detail_train"],
#     split_col_names=[
#         [("COUPON_ID_hash", "COUPON_ID_hash")],
#         [("USER_ID_hash", "USER_ID_hash")],
#     ]
# )

coupon_detail_folds = utils.split_k_fold_on_parent(
    user_folds, original_data["coupon_detail_train"], [("USER_ID_hash", "USER_ID_hash")])

# %%
# coupon_visit_folds = utils.split_k_fold_on_multiple_parents(
#     parents_folds=[coupon_folds, user_folds],
#     child_table=original_data["coupon_visit_train"],
#     split_col_names=[
#         [("VIEW_COUPON_ID_hash", "COUPON_ID_hash")],
#         [("USER_ID_hash", "USER_ID_hash")],
#     ]
# )

coupon_visit_folds = utils.split_k_fold_on_parent(
    user_folds, original_data["coupon_visit_train"], [("USER_ID_hash", "USER_ID_hash")])

# %%
utils.save_folds(
    # coupon_folds, coupon_area_folds,
    [user_folds, coupon_detail_folds, coupon_visit_folds],
    dataset_name,
    ["user_list", "coupon_detail_train", "coupon_visit_train"])

# %%
