# %%
import os
from dotenv import load_dotenv
from gretel_trainer.relational import *
from gretel_trainer.relational import RelationalData
import pandas as pd
from tqdm import tqdm
from IPython.display import display, HTML
from gretel_trainer.relational import MultiTable
from rike import utils
from gretel_client import configure_session
import argparse

# %%
# get api key from .env
load_dotenv()

configure_session(api_key=os.environ.get("GRETEL_API_KEY"), validate=True)

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str,
                  default="telstra-competition-dataset")
args.add_argument("--start_fold", type=int, default=0)
args.add_argument("--end_fold", type=int, default=5)
args = args.parse_args()

DATASET_NAME = args.dataset_name
METHOD_NAME = 'gretel'

# %%


def join_tables(parent: str, child: str, relational_data=RelationalData()):
    p_key = relational_data.get_primary_key(parent)
    f_key = ""
    for fk in relational_data.get_foreign_keys(child):
        if fk.parent_table_name == parent:
            f_key = fk.column_name
        else:
            logging.warning(
                "The input parent and child table must be related.")

    parent_df = relational_data.get_table_data(parent)
    child_df = relational_data.get_table_data(child)

    joined_data = child_df.merge(
        parent_df, how="left", left_on=p_key, right_on=f_key)

    print(f"Number of records in {child} table:\t {len(child_df)}")
    print(f"Number of records in {parent} table:\t {len(parent_df)}")
    print(f"Number of records in joined data:\t {len(joined_data)}")

    return joined_data.head()

# %%
# @title
# Alternatively, manually define relational data
# Uncomment code to run


limit = args.end_fold
for k in tqdm(range(args.start_fold, limit)):
    csv_dir = f"../../data/splits/{DATASET_NAME}/{DATASET_NAME}_leave_out_{k}"
    parent_table = (f"train_train_{k}", "id")
    child1_table = (f"severity_type_train_{k}", "id")
    child2_table = (f"resource_type_train_{k}", "id")
    child3_table = (f"log_feature_train_{k}", "id")
    child4_table = (f"event_type_train_{k}", "id")
    tables = [
        # ("table_name", "primary_key")
        parent_table,
        child1_table,
        child2_table,
        child3_table,
        child4_table
    ]

    foreign_keys = [
        # ("fkey_table.fkey", "pkey_table.pkey")
        (f"{child1_table[0]}.{parent_table[1]}",
            f"{parent_table[0]}.{parent_table[1]}"),
        (f"{child2_table[0]}.{parent_table[1]}",
            f"{parent_table[0]}.{parent_table[1]}"),
        (f"{child3_table[0]}.{parent_table[1]}",
            f"{parent_table[0]}.{parent_table[1]}"),
        (f"{child4_table[0]}.{parent_table[1]}",
            f"{parent_table[0]}.{parent_table[1]}")
    ]

    relational_data = RelationalData()

    tables_train, tables_test = utils.get_train_test_split(DATASET_NAME, k)

    for table, pk in tables:
        relational_data.add_table(
            name=table, primary_key=pk, data=tables_train[table.split("_train")[0]])

    for fk, ref in foreign_keys:
        relational_data.add_foreign_key(foreign_key=fk, referencing=ref)

    # print("\033[1m Source Data: \033[0m")
    # source_data = join_tables(parent_table[0], child1_table[0], relational_data=relational_data)

    gretel_model = "lstm"
    multitable = MultiTable(
        relational_data,
        project_display_name=f"Synthesize {DATASET_NAME} - {gretel_model}",
        gretel_model=gretel_model,
        # refresh_interval=60
    )
    multitable.train()
    parent_table_name = parent_table[0].split("_train")[0]
    ratio = tables_test[parent_table_name].shape[0] / \
        tables_train[parent_table_name].shape[0]
    multitable.generate(record_size_ratio=ratio)

    synthetic_data = multitable.synthetic_output_tables
    utils.save_data(synthetic_data, DATASET_NAME, k, method=METHOD_NAME)

# %%
# METHOD_NAME='gretel'
# synthetic_data = multitable.synthetic_output_tables
# utils.save_data(synthetic_data, DATASET_NAME, k, method=METHOD_NAME)
# %%
