import os
import pandas as pd
from realtabformer import REaLTabFormer

parent_df = pd.read_csv("/content/drive/MyDrive/MAG-1/DS-Project/colab/Rossman_data/source/store.csv")
child_df = pd.read_csv("/content/drive/MyDrive/MAG-1/DS-Project/colab/Rossman_data/source/sales.csv")

join_on = "Store"
assert ((join_on in parent_df.columns) and
        (join_on in child_df.columns))

parent_model = REaLTabFormer(model_type="tabular")
parent_model_path = 'models/realtabformer_rossmann_store_sales' # TODO: fix this path

parent_model = REaLTabFormer(
    model_type="tabular",
    parent_realtabformer_path=parent_model_path,
)
parent_model = parent_model.load_from_dir(parent_model_path)
parent_model.fit(parent_df.drop(join_on, axis=1))
parent_samples = parent_model.sample(len(parent_df))

parent_samples.to_csv(f'data/synthetic/rossmann-store-sales/REaLTabFormer/rossmann_store_fold_0.csv', index=False) # TODO: fix this path

child_model = REaLTabFormer(
    model_type="relational",
    parent_realtabformer_path=parent_model_path,
    output_max_length=None,
    train_size=0.95,
    batch_size=1)

child_model.fit(
    df=child_df.sample(n=10000, random_state=42),
    df = child_df,
    in_df=parent_df,
    join_on=join_on)