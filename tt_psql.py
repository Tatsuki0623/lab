import os
import psycopg
from psycopg.rows import dict_row
from concurrent.futures import ThreadPoolExecutor

# ソースDBとターゲットDBの接続設定
SOURCE_DB_CONFIG = {
    'dbname': 'source_db',
    'user': 'test_user',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

TARGET_DB_CONFIG = {
    'dbname': 'target_db',
    'user': 'test_user',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

def execute_insert(connection, query, value):
    retunr_val = None
    try:
        with connection.cursor() as cur:
            if "batch_result" in query:
                cur.executemany(query,value)
            elif "batch" in query:
                cur.execute(query, value)
                retunr_val = cur.fetchone()["id"]
    except (psycopg.OperationalError, psycopg.errors.UniqueViolation, Exception) as e:
        print(e)
    return retunr_val


def total():
    with psycopg.connect(**SOURCE_DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM batch")
            total_rows = cur.fetchone()["count"]
    return total_rows

def fetch_batches(offset, limit):
    """
    ソースデータベースから指定範囲のデータを取得（OFFSETとLIMITを使用）
    """
    with psycopg.connect(**SOURCE_DB_CONFIG, row_factory = dict_row) as conn:
        with conn.cursor() as cur:
            # batchデータ取得
            cur.execute(
                "SELECT * FROM batch ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset)
            )
            batch_data = cur.fetchall()
    return batch_data

def fetch_batche_results(batch_id):
    """
    ソースデータベースから指定範囲のデータを取得（OFFSETとLIMITを使用）
    """
    with psycopg.connect(**SOURCE_DB_CONFIG, row_factory = dict_row) as conn:
        with conn.cursor() as cur:
            # batchデータ取得
            cur.execute(
                f"SELECT * FROM batch_result where batch_id = {batch_id}"
            )
            batch_result_data = cur.fetchall()
    return batch_result_data

def insert_batch(batches):
    connection = psycopg.connect(**TARGET_DB_CONFIG, row_factory = dict_row)

    for batch in batches:
        batch_col = list(batch.keys())
        old_id = batch["id"]
        batch_col.remove('id')
        join_batches_col = ",".join(batch_col)

        batch.pop('id')
        batch_val = tuple(batch.values())

        insert_query_batch = f"insert into batch ({join_batches_col}) values({",".join(['%s'] * len(batch_col))}) RETURNING id"
        new_id = execute_insert(connection, insert_query_batch, batch_val)

        batch_results = fetch_batche_results(old_id)
        if len(batch_results) == 0:
            continue

        batch_results_col = batch_results[0].keys()
        join_batch_results_col = ",".join(batch_results_col)

        batch_results_val = []
        for batch_result in batch_results:
            batch_result["batch_id"] = new_id
            new_batch_reslut = tuple(batch_result.values())
            batch_results_val.append(new_batch_reslut)
        
        insert_query_batch_result = f"insert into batch_result ({join_batch_results_col}) values({",".join(['%s'] * len(batch_results_col))})"
        execute_insert(connection, insert_query_batch_result, batch_results_val)
    
    return None
