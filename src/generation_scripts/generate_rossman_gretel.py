# %%
from gretel_trainer.relational import *
from gretel_trainer.relational import RelationalData
import pandas as pd
from tqdm import tqdm
from IPython.display import display, HTML
from gretel_trainer.relational import MultiTable
from rike.generation import generation_utils

DATASET_NAME = "rossmann-store-sales"
METHOD_NAME='gretel'

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


for k in tqdm(range(10)):
    csv_dir = f"../../data/splits/{DATASET_NAME}/{DATASET_NAME}_leave_out_{k}"
    parent_table = (f"store_train_{k}", "Store")
    child1_table = (f"test_train_{k}", "Id")
    child1_table_fk = parent_table[1]
    tables = [
        # ("table_name", "primary_key")
        parent_table,
        child1_table,
    ]

    foreign_keys = [
        # ("fkey_table.fkey", "pkey_table.pkey")
        (f"{child1_table[0]}.{child1_table_fk}",
         f"{parent_table[0]}.{parent_table[1]}"),
    ]

    relational_data = RelationalData()

    for table, pk in tables:
        relational_data.add_table(
            name=table, primary_key=pk, data=pd.read_csv(f"{csv_dir}/{table}.csv"))

    for fk, ref in foreign_keys:
        relational_data.add_foreign_key(foreign_key=fk, referencing=ref)

    print("\033[1m Source Data: \033[0m")
    source_data = join_tables(parent_table[0], child1_table[0], relational_data=relational_data)

    gretel_model = "lstm"
    multitable = MultiTable(
        relational_data,
        project_display_name=f"Synthesize {DATASET_NAME} - {gretel_model}",
        gretel_model=gretel_model,
        # refresh_interval=60
    )
    multitable.train()
    multitable.generate(record_size_ratio=0.1)

    table = "store"  # @param {type:"string"}

    # source_table = multitable.relational_data.get_table_data(table).head(5)
    # synth_table = multitable.synthetic_output_tables[table][source_table.columns].head(
    #     5)
    # print("\033[1m Source Table:")
    # display(source_table)
    # print("\n\n\033[1m Synthesized Table:")
    # display(synth_table)

    synthetic_data = multitable.synthetic_output_tables
    generation_utils.save_data(synthetic_data, DATASET_NAME, k, method=METHOD_NAME)
    break

# %%
METHOD_NAME='gretel'
synthetic_data = multitable.synthetic_output_tables
generation_utils.save_data(synthetic_data, DATASET_NAME, k, method=METHOD_NAME)
# %%
