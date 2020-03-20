#
# Database tools.
#
# Module includes tools for reading and loading data, 
# executing SQL queries for various databases.
# Currently realized: SQLite, Teradata, Hive 
#

import warnings
import pandas as pd
import sqlite3
import turbodbc
import puretransport
from pyhive import hive
import csv
import re
import os
import shutil
from tqdm import tqdm
warnings.filterwarnings('ignore')


class SQLiteDB:
    def __init__(self, database):
        """SQLite database initialization.
        
        Parameters:
            database (str): path to database file.
        """
        self.database = database
        
    def load(self, filename, table, primary_key):
        """Load data from file to SQLite database.
        
        Parameters:
            filename (str): path to file
            table (str): table name
            primary_key (str): column names for primary key ('id' or 'id1, id2, id3')
        """
        con = sqlite3.connect(self.database)
        cur = con.cursor()
        
        df = pd.read_csv(filename, nrows=5)
        dtypes = pd.Series(df.dtypes.astype(str).tolist(), index=df.columns.tolist())
        type_map = {'object': 'text', 'int': 'integer', 'float': 'real'}
        dtypes = dtypes.apply(lambda x: re.sub(r'[0-9]+', '', x))
        dtypes = dtypes.map(type_map)
        
        query = "create table if not exists {} ({}, primary key ({}))".format(
            table,
            ", ".join([" ".join(i) for i in zip(dtypes.index, dtypes.values)]),
            primary_key
        )
        cur.execute(query)

        with open(filename, 'rt') as f:
            next(f, None)
            reader = csv.reader(f)
            for line in tqdm(reader):
                cur.execute("insert or replace into {} values ({});".format(
                    table, ", ".join(list("?" * len(dtypes)))), line
                )

        con.commit()
        con.close()
    
    def read(self, query):
        """Read data from SQLite database.
        
        Parameters:
            query (str): SQL query
            
        Returns:
            pandas.DataFrame with results of query
        """
        con = sqlite3.connect(self.database)
        cur = con.cursor()
        cur.execute(query)
        fetched_cols = [i[0] for i in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=fetched_cols)
        con.close()
        return df
    
    def execute(self, query):
        """Execute SQL query on SQLite database.
        
        Parameters:
            query (str): SQL query
        """
        con = sqlite3.connect(self.database)
        cur = con.cursor()
        cur.execute(query)
        con.commit()
        con.close()
        
        
class TeradataDB:
    def __init__(self, dsn, login, password, ip, database, 
                 fastload_path, temp_folder):
        """Teradata database initialization.
        
        Parameters:
            dsn (str): ODBC connection name
            login (str): login
            password (str): password
            ip (str): ip
            database (str): database for loading new data
            fastload_path (str): path to fastload utility
            temp_folder (str): folder name for saving temporary files
            fastload_err1 (str): temporary table for fastload errors 1
            fastload_err2 (str): temporary table for fastload errors 2
        """
        self.dsn = dsn
        self.login = login
        self.password = password
        self.ip = ip
        self.database = database
        self.fastload_path = fastload_path
        self.temp_folder = temp_folder
        self.fastload_err1 = fastload_err1
        self.fastload_err2 = fastload_err2
        
    def load(self, df, table):
        """Load pandas.DataFrame to Teradata table.
        
        Parameters:
            df (pandas.DataFrame): DataFrame for loading
            table (str): table name
        """
        con = turbodbc.connect(
            dsn=self.dsn, 
            turbodbc_options=turbodbc.make_options(prefer_unicode=True, autocommit=True)
        )
        cur = con.cursor()
        
        try:
            drop_sql = 'drop table {}.{}'.format(self.database, table)
            cur.execute(drop_sql)
        except turbodbc.DatabaseError:
            pass
            
        teradata_types = {
            'int8': 'byteint', 'int16': 'smallint', 'int32': 'integer', 'int64': 'bigint',
            'float16': 'float', 'float32': 'float', 'float64': 'double', 'object': 'varchar',
            'bool': 'byteint'
        }
        
        query = 'create multiset table {}.{} ('.format(self.database, table)
        for idx, dtype in zip(df.dtypes.index, df.dtypes.values):
            dtype = str(dtype)
            td_type = teradata_types[dtype] + \
                        ('' if dtype != 'object' else '({})'.format(df[idx].str.len().max()))
            query += '{} {}, '.format(idx, td_type)
        query = query[:-2] + ') no primary index'
        
        cur.execute(query)
        con.close()
        
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        df.to_csv('{}/df.csv'.format(self.temp_folder), sep=',', decimal='.', index=False)
        
        script = \
        'set session charset "UTF8";\n' + \
        f'logon {self.teradata_ip}/{self.login}, {self.password};\n' + \
        f'database {self.database};\n' + \
        f'begin loading {self.table}\n' + \
        f'errorfiles {self.fastload_err1}, {self.fastload_err2}\n' + \
        'checkpoint 1000000;\n' + \
        'set record vartext "," nostop;\n' + \
        'record 2;\n' + \
        'define\n' + \
        ',\n'.join(
            ['{} (varchar({}))'.format(col, df[col].astype(str).str.len().max()) for col in df.columns]
        ) + '\n' + \
        'file = {};\n'.format(os.getcwd() + '\\{}\\df.csv'.format(self.temp_folder)) + \
        f'insert into {table}\n' + \
        'values(\n' + \
        ',\n'.join([':' + col for col in df.columns]) + \
        ');\n' + \
        'end loading;\n' + \
        'logoff;\n' + \
        'quit;'
        script_file = open(f'{self.temp_folder}/fastload_script.txt', "w+")
        script_file.write(script)
        script_file.close()
        command = f'cd {self.fastload_path} | fastload.exe < ' + \
                  f'{os.getcwd()}\\{self.temp_folder}\\fastload_script.txt'
        flg = os.system(command)
        shutil.rmtree(self.temp_folder)
        
    def read(self, query):
        """Read data from Teradata database.
        
        Parameters:
            query (str): SQL query
            
        Returns:
            pandas.DataFrame with results of query
        """
        con = turbodbc.connect(
            dsn=self.dsn, turbodbc_options=turbodbc.make_options(prefer_unicode=True, autocommit=True)
        )
        cur = con.cursor()
        cur.execute(query)
        df = pd.DataFrame(cur.fetchallnumpy())
        df.columns = df.columns.str.lower()
        con.close()
        return df      
        
    def execute(self, query):
        """Execute SQL query on Teradata database.
        
        Parameters:
            query (str): SQL query
        """
        con = turbodbc.connect(
            dsn=self.dsn, turbodbc_options=turbodbc.make_options(prefer_unicode=True, autocommit=True)
        )
        cur = con.cursor()
        cur.execute(query)
        con.close()
        
        
class HiveDB:
    def __init__(self, username, password, host, port):
        """Hive database initialization.
        
        Parameters:
            username (str): username
            password (str): password
            host (str): host
            port (str): port
        """
        self.login = login
        self.host = host
        self.port = port
        
    def read(self, query):
        """Read data from Hive database.
        
        Parameters:
            query (str): SQL query

        Returns:
            pandas.DataFrame with results of query
        """
        transport = puretransport.transport_factory(
            host=self.host, port=self.port, username=self.username, password=self.password
        )
        con = hive.connect(thrift_transport=transport)
        cur = con.cursor()
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns = [x[0] for x in cur.description])
        df.columns = [i[i.rfind('.')+1:] for i in df.columns.tolist()]
        con.close()
        return df