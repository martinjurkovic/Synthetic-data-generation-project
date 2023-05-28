import os
import argparse

import psycopg2
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from rike import utils
from rike.generation import sdv_metadata
from psycopg2.extras import execute_values
from io import StringIO

load_dotenv()

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="mutagenesis")
args.add_argument("--varchar-length", type=int, default=255)
args.add_argument("--pk-length", type=int, default=20)
args.add_argument("--drop", type=bool, default=True)
args = args.parse_args()

dataset_name = args.dataset_name

# Connect to the database
connection = psycopg2.connect(host=os.environ.get('PG_HOST'),
                        port=os.environ.get('PG_PORT'),
                        user=os.environ.get('PG_USER'),
                        password=os.environ.get('PG_PASSWORD'),
                        # dbname=args.dataset_name.split("-")[0],
                        dbname="rossmann",
                        sslmode='require')

cursor = connection.cursor()


# TABLE GENERATION FUNCTIONS
def find_fk(parent, reference, metadata):
    for field in metadata.to_dict()['tables'][parent]['fields']:
        # remove all digits from the field name
        reference_digit = ''.join([i for i in reference if not i.isdigit()])
        if field in reference or field in reference_digit:
            return field


def get_foreign_key_reference(table_name, field, metadata):
    parents = metadata.get_parents(table_name)
    for parent in parents:
        fks = metadata.get_foreign_keys(parent, table_name)
        for fk in fks:
            if fk == field:
                return parent, find_fk(parent, field, metadata)
    return None, None


def create_table_query(table_name, metadata, k=0, varchar_length=255, pk_length=20):
    fields = metadata.to_dict()['tables'][table_name]['fields']
    fields_str = ''
    for field, values in fields.items():
        if values['type'] == 'id':
            if values['subtype'] == 'integer':
                fields_str += f"{field} INTEGER"
            elif values['subtype'] == 'string':
                fields_str += f"{field} VARCHAR({pk_length})"
            parent, parent_field = get_foreign_key_reference(table_name, field, metadata)
            # check if its the primary key
            if metadata.to_dict()['tables'][table_name]['primary_key'] == field:
                fields_str += ' PRIMARY KEY, '
            elif parent is not None:
                fields_str += f" REFERENCES {parent}_fold_{k}({parent_field}), "
            
        elif values['type'] == 'categorical':
            fields_str += f"{field} VARCHAR({varchar_length}), "
        elif values['type'] == 'numerical':
            fields_str += f"{field} FLOAT, "
        elif values['type'] == 'boolean':
            fields_str += f"{field} BOOLEAN, "
        elif values['type'] == 'datetime':
            fields_str += f"{field} DATE, "
        else:
            raise ValueError(f'Unknown type {values["type"]} for field {field}')
    fields_str = fields_str[:-2]
    query = f"CREATE TABLE IF NOT EXISTS {table_name}_fold_{k} ({fields_str});"
    return query


def execute_query(query, cursor, connection):
    try:
        # Execute the query
        cursor.execute(query)

        # Commit the transaction
        connection.commit()

    except psycopg2.Error as error:
        # Handle any errors that occur during query execution
        print("Error executing query:", error)


def execute_write_query(query, cursor, connection, values):
        try:
            # Execute the INSERT query with %s placeholder
            cursor.execute(query, values)
    
            # Commit the transaction
            connection.commit()
    
            #print("Query executed successfully")
    
        except psycopg2.Error as error:
            raise error
            # Handle any errors that occur during query execution
            # print("Error executing query:", error)


def insert_rows(table_name, df, k=0):
    insert_query = f"INSERT INTO {table_name}_fold_{k} VALUES (" + "%s,"*(len(df.columns)-1) + "%s)"
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        execute_write_query(insert_query, cursor, connection, tuple(row))
        # break

# psycopg2.extensions.register_adapter(float, lambda x: 'NULL' if pd.isna(x) else float(x))        
def insert_batch_rows(table_name, df, k=0, batch_size=100):
    insert_query = f"INSERT INTO {table_name}_fold_{k} VALUES (" + "%s,"*(len(df.columns)-1) + "%s)"
    rows = [tuple(row) for _, row in df.iterrows()]
    # Insert rows in batches
    total_rows = len(rows)
    batch_size = total_rows if batch_size > total_rows else batch_size
    for i in tqdm(range(0, total_rows, batch_size)):
        batch = rows[i:i+batch_size] if i+batch_size <= total_rows else rows[i:]
        cursor.executemany(insert_query, batch)
        connection.commit()
        # break

def insert_batch_rows_zurich(table_name, df, k=0, batch_size=100):
    # Convert NaN values to None
    df = df.where(pd.notnull(df), None)
    
    # Create a buffer to hold the data
    buffer = StringIO()
    df.to_csv(buffer, sep='\t', header=False, index=False, na_rep='NaN')
    buffer.seek(0)

    # Connect to the database and execute the query
    with connection as conn:
        with conn.cursor() as cursor:
            cursor.copy_from(buffer, f"{table_name}_fold_{k}", null="NaN", columns=df.columns, sep='\t')
            conn.commit()


if __name__ == "__main__":
    # CREATE TABLES
    # clear the database
    if args.drop:
        execute_query("DROP SCHEMA public CASCADE; CREATE SCHEMA public;", cursor, connection)

    # create the tables
    for k in tqdm(range(10)):
        tables_train, tables_test = utils.get_train_test_split(dataset_name, k)
        if dataset_name in ("zurich_mle", "zurich"):
            tables_train['claims']['customer_id'] = tables_train['claims']['customer_id'].astype(int)
            tables_train['claims']['claim_id'] = tables_train['claims']['claim_id'].astype(int)
            tables_train['claims']['policy_id'] = tables_train['claims']['policy_id'].astype(int)
            tables_train['policies']['customer_id'] = tables_train['policies']['customer_id'].astype(int)
        if dataset_name == "zurich_mle":
            tables_train = tables_test.copy()
        metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
        for table_name in metadata.to_dict()['tables'].keys():
            # create the table
            query = create_table_query(table_name, metadata, k, 
                                    varchar_length=args.varchar_length, 
                                    pk_length=args.pk_length)
            execute_query(query, cursor, connection)
            # check if the table exists
            # insert the rows
            # connection.commit()
            fields = metadata.to_dict()['tables'][table_name]['fields']
            # reorder the columns of tables_train[table_name] to be like in the metadata
            tables_train[table_name] = tables_train[table_name][list(fields.keys())]
            # insert_rows(table_name, tables_train[table_name], k)
            if dataset_name in ("zurich_mle", "zurich"):
                insert_batch_rows_zurich(table_name, tables_train[table_name], k, batch_size=1000)
            else:
                insert_batch_rows(table_name, tables_train[table_name], k, batch_size=1000)
