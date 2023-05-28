import os
import mysql.connector


DATA_DIR = 'data/original'
def download_dataset(dataset):
    print(f'Downloading {dataset} dataset')

    # create a directory for the dataset if it doesn't exist
    if not os.path.exists(f'{DATA_DIR}/{dataset.lower()}'):
        os.makedirs(f'{DATA_DIR}/{dataset.lower()}')

    # connect to the MariaDB database
    mydb = mysql.connector.connect(
    host="relational.fit.cvut.cz",
    port=3306,
    user="guest",
    password="relational",
    database=dataset
    )

    mycursor = mydb.cursor()

    mycursor.execute("SHOW TABLES")
    tables = [table[0] for table in mycursor.fetchall()]


    # for each table in the database fetch all rows and save them to a csv file
    for table in tables:

        # fetch columns
        mycursor.execute(f'SHOW COLUMNS FROM `{table}`')
        columns = [column[0] for column in mycursor.fetchall()]
        print(f'Fetching data from {table: <8} table', end='... ')
        query = f'SELECT * FROM `{table}`'
        mycursor.execute(query)
        rows = mycursor.fetchall()
        with open(f'{DATA_DIR}/{dataset.lower()}/{table}.csv', "w") as f:
            f.write(','.join(columns) + '\n')
            for row in rows:
                f.write(','.join([str(r) for r in row]) + '\n')
        print('Done')



if __name__ == '__main__':
    download_dataset('Biodegradability')
    download_dataset('mutagenesis')