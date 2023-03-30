import os
import pandas as pd
from sdmetrics.reports import utils

def read_tables(path, split_by="-", name_index=-1, **kwargs):
    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_name = file[:-4].split(split_by)[name_index]
            table = pd.read_csv(
                path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
            tables[table_name] = table
    return tables


def plot_all_columns(real_data, synthetic_data, metadata):
    for table_name, table in real_data.items():
        for column_name in table.columns:
            try:
                fig = utils.get_column_plot(
                    real_data=real_data[table_name],
                    synthetic_data=synthetic_data[table_name],
                    column_name=column_name,
                    metadata=metadata['tables'][table_name]
                )
                fig.show()
            except Exception as e:
                print(f"Could not plot {table_name}.{column_name}")
                print(e)
