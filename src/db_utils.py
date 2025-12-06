import pandas as pd
import psycopg2
import logging

def create_table_if_not_exists(conn, schema_name,table_name: str, df: pd.DataFrame, primary_key: str):
    """
    DataFrame の列名・型情報を用いて、指定した table_name が存在しない場合に
    CREATE TABLE するヘルパー関数。
    """
    # テーブルの存在確認
    with conn.cursor() as cursor:
        check_table_sql = f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{table_name}' and table_schema = '{schema_name}'
        """
        cursor.execute(check_table_sql, (table_name,))
        table_count = cursor.fetchone()[0]

    if table_count == 0:
        # テーブルが存在しない場合 -> CREATE TABLE

        # DataFrame の列名を取り出し
        columns = list(df.columns)

        # Pandas dtype -> PostgreSQL の型マッピング (簡易版)
        def map_dtype_to_pg(dtype):
            """
            必要に応じてマッピングをカスタマイズしてください
            """
            if pd.api.types.is_datetime64_any_dtype(dtype):
                return "timestamp NULL"
            else:
                return "varchar(100)"  # 文字列などは TEXT で簡易対応

        # CREATE TABLE 文を組み立てる
        column_definitions = []
        for col in columns:
            col_dtype = df[col].dtype
            pg_type = map_dtype_to_pg(col_dtype)

            # 主キーの場合は "PRIMARY KEY" を付与
            if col == primary_key:
                column_definitions.append(f"{col} {pg_type} PRIMARY KEY")
            else:
                column_definitions.append(f"{col} {pg_type}")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            {', '.join(column_definitions)}
        );
        """

        with conn.cursor() as cursor:
            cursor.execute(create_table_sql)
            conn.commit()

        logging.info(f"Table '{table_name}.{table_name}' did not exist. Created a new table.")

def add_new_columns_to_table(conn, schema_name,table_name: str, df: pd.DataFrame):
    """
    DataFrame の列情報をもとに、テーブルに新しい列が足りない場合に
    ALTER TABLE ADD COLUMN を実行する関数。
    """
    cursor = conn.cursor()

    try:
        # 既存のテーブルカラム情報を取得
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' and table_schema = '{schema_name}'")
        existing_columns_info = cursor.fetchall()
        existing_columns = [col[0] for col in existing_columns_info]

        # DataFrame のカラムと比較
        columns_to_add = []
        for col in df.columns:
            if col not in existing_columns:
                columns_to_add.append(col)

        if columns_to_add:
            logging.info(f"New columns detected: {columns_to_add}. Adding them to table '{schema_name}.{table_name}'.")

            # Pandas dtype -> PostgreSQL の型マッピング (create_table_if_not_exists と共通)
            def map_dtype_to_pg(dtype):
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    return "timestamp NULL"
                else:
                    return "varchar(100)"  # 文字列などは TEXT で簡易対応

            # ALTER TABLE 文を組み立て、新しいカラムを追加
            for col in columns_to_add:
                col_dtype = df[col].dtype
                pg_type = map_dtype_to_pg(col_dtype)
                alter_table_sql = f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN {col} {pg_type};"
                cursor.execute(alter_table_sql)
            conn.commit() #  忘れずにコミット
            logging.info(f"Successfully added new columns to table '{schema_name}.{table_name}'.")
        else:
            logging.info(f"No new columns to add for table '{schema_name}.{table_name}'.")

    except Exception as e:
        conn.rollback() # エラー発生時はロールバック
        logging.error(f"Error occurred while adding new columns: {e}")
        raise e # エラーを再Raiseして、呼び出し元に伝える

    finally:
        cursor.close()
        
def upsert_dataframe_to_postgresql(
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    primary_key: str,
    conn_params: dict
):
    """
    DataFrame の行を PostgreSQL のテーブルに upsert (INSERT or UPDATE) する。
    
    :param df: upsert したいデータを持つ Pandas DataFrame
    :param table_name: 対象のテーブル名
    :param primary_key: ON CONFLICT に指定する主キー
    :param conn_params: psycopg2.connect に渡すための接続パラメータ(辞書)
    例) {"dbname": "test_db", "user": "postgres", "password": "****", "host": "localhost", "port": 5432}
    """
    # カラム一覧をリストとして取得
    columns = list(df.columns)

    # INSERT 用のプレースホルダ（PostgreSQL では '%s' を使う）
    placeholders = ", ".join(["%s"] * len(columns))

    # UPDATE するカラム (primary_key 以外) のリストを作成
    # EXCLUDED で新しい値を参照
    update_columns = ", ".join([
        f"{col} = EXCLUDED.{col}" for col in columns if col != primary_key
    ])

    # INSERT 文の組み立て
    # ON CONFLICT (主キー) DO UPDATE SET ...
    query = f"""
    INSERT INTO {schema_name}.{table_name} ({', '.join(columns)})
    VALUES ({placeholders})
    ON CONFLICT ({primary_key}) DO UPDATE
    SET {update_columns};
    """

    try:
        # psycopg2 で PostgreSQL に接続
        with psycopg2.connect(**conn_params) as conn:
            # テーブルが存在しなければ作成
            create_table_if_not_exists(conn, schema_name,table_name, df, primary_key)
            add_new_columns_to_table(conn, schema_name,table_name, df)

            # psycopg2 で PostgreSQL に接続
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cursor:
                    # 行ごとにデータを upsert
                    for _, row in df.iterrows():
                        cursor.execute(query, tuple(row))

        # コミット
        conn.commit()

        logging.info(f"Data successfully upserted into '{schema_name}'.'{table_name}'.")
    
    except Exception as e:
        logging.error(f"Error occurred during upsert: {e}")

def create_table_if_not_exists_multi_pk(conn, schema_name, table_name: str, df: pd.DataFrame, primary_keys: list):
    """
    DataFrame の列名・型情報を用いて、指定した table_name が存在しない場合に
    CREATE TABLE するヘルパー関数。（複数プライマリキー対応版）
    """
    # テーブルの存在確認
    with conn.cursor() as cursor:
        check_table_sql = f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{table_name}' and table_schema = '{schema_name}'
        """
        cursor.execute(check_table_sql)
        table_count = cursor.fetchone()[0]

    if table_count == 0:
        # テーブルが存在しない場合 -> CREATE TABLE

        # DataFrame の列名を取り出し
        columns = list(df.columns)

        # Pandas dtype -> PostgreSQL の型マッピング (簡易版)
        def map_dtype_to_pg(dtype):
            """
            必要に応じてマッピングをカスタマイズしてください
            """
            if pd.api.types.is_datetime64_any_dtype(dtype):
                return "timestamp NULL"
            else:
                return "varchar(100)"  # 文字列などは TEXT で簡易対応

        # CREATE TABLE 文を組み立てる
        column_definitions = []
        for col in columns:
            col_dtype = df[col].dtype
            pg_type = map_dtype_to_pg(col_dtype)
            column_definitions.append(f"{col} {pg_type}")

        # 複数プライマリキーの制約を追加
        if primary_keys:
            pk_str = ", ".join(primary_keys)
            column_definitions.append(f"PRIMARY KEY ({pk_str})")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            {', '.join(column_definitions)}
        );
        """

        with conn.cursor() as cursor:
            cursor.execute(create_table_sql)
            conn.commit()

        logging.info(f"Table '{schema_name}.{table_name}' did not exist. Created a new table with multi-PK.")

def add_new_columns_to_table(conn, schema_name, table_name: str, df: pd.DataFrame):
     """
     DataFrame の列情報をもとに、テーブルに新しい列が足りない場合に
     ALTER TABLE ADD COLUMN を実行する関数。
     """
     cursor = conn.cursor()

     try:
         # 既存のテーブルカラム情報を取得
         cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' and table_schema = '{schema_name}'")
         existing_columns_info = cursor.fetchall()
         existing_columns = [col[0] for col in existing_columns_info]

         # DataFrame のカラムと比較
         columns_to_add = []
         for col in df.columns:
             if col not in existing_columns:
                 columns_to_add.append(col)

         if columns_to_add:
             logging.info(f"New columns detected: {columns_to_add}. Adding them to table '{schema_name}.{table_name}'.")

             # Pandas dtype -> PostgreSQL の型マッピング (create_table_if_not_exists と共通)
             def map_dtype_to_pg(dtype):
                 if pd.api.types.is_datetime64_any_dtype(dtype):
                     return "timestamp NULL"
                 else:
                     return "varchar(100)"  # 文字列などは TEXT で簡易対応

             # ALTER TABLE 文を組み立て、新しいカラムを追加
             for col in columns_to_add:
                 col_dtype = df[col].dtype
                 pg_type = map_dtype_to_pg(col_dtype)
                 alter_table_sql = f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN {col} {pg_type};"
                 cursor.execute(alter_table_sql)
             conn.commit() #  忘れずにコミット
             logging.info(f"Successfully added new columns to table '{schema_name}.{table_name}'.")
         else:
             logging.info(f"No new columns to add for table '{schema_name}.{table_name}'.")

     except Exception as e:
         conn.rollback() # エラー発生時はロールバック
         logging.error(f"Error occurred while adding new columns: {e}")
         raise e # エラーを再Raiseして、呼び出し元に伝える

     finally:
         cursor.close()

def upsert_dataframe_to_postgresql_multi_pk(
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    primary_keys: list,
    conn_params: dict
):
    """
    DataFrame の行を PostgreSQL のテーブルに upsert (INSERT or UPDATE) する。
    複数プライマリキーに対応。
    
    :param df: upsert したいデータを持つ Pandas DataFrame
    :param table_name: 対象のテーブル名
    :param primary_keys: ON CONFLICT に指定する主キーのリスト (list of str)
    :param conn_params: psycopg2.connect に渡すための接続パラメータ(辞書)
    """
    # カラム一覧をリストとして取得
    columns = list(df.columns)

    # INSERT 用のプレースホルダ
    placeholders = ", ".join(["%s"] * len(columns))

    # UPDATE するカラム (primary_keys 以外) のリストを作成
    update_columns_list = [
        f"{col} = EXCLUDED.{col}" for col in columns if col not in primary_keys
    ]
    update_columns = ", ".join(update_columns_list)

    # ON CONFLICT のキー部分を作成
    conflict_target = ", ".join(primary_keys)

    # INSERT 文の組み立て
    # ON CONFLICT (pk1, pk2...) DO UPDATE SET ...
    query = f"""
    INSERT INTO {schema_name}.{table_name} ({', '.join(columns)})
    VALUES ({placeholders})
    ON CONFLICT ({conflict_target}) DO UPDATE
    SET {update_columns};
    """

    try:
        # psycopg2 で PostgreSQL に接続
        with psycopg2.connect(**conn_params) as conn:
            # テーブルが存在しなければ作成
            create_table_if_not_exists_multi_pk(conn, schema_name, table_name, df, primary_keys)
            add_new_columns_to_table(conn, schema_name, table_name, df)

            # psycopg2 で PostgreSQL に接続
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cursor:
                    # 行ごとにデータを upsert
                    for _, row in df.iterrows():
                        cursor.execute(query, tuple(row))
        
        # コミット
        conn.commit()

        logging.info(f"Data successfully upserted into '{schema_name}'.'{table_name}' with multi-PK.")
    
    except Exception as e:
        logging.error(f"Error occurred during upsert: {e}")
