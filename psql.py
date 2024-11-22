import os
import psycopg
from concurrent.futures import ThreadPoolExecutor

# ソースDBとターゲットDBの接続設定
SOURCE_DB_CONFIG = {
    'dbname': 'source_database',
    'user': 'source_user',
    'password': 'source_password',
    'host': 'localhost',
    'port': 5432
}

TARGET_DB_CONFIG = {
    'dbname': 'target_database',
    'user': 'target_user',
    'password': 'target_password',
    'host': 'localhost',
    'port': 5432
}

# バッチ設定
BATCH_SIZE = 1000  # 1回のバッチサイズ
PROGRESS_FILE = "progress_offset.log"  # 進捗管理用ログファイル


def fetch_batches(offset, limit):
    """
    ソースデータベースから指定範囲のデータを取得（OFFSETとLIMITを使用）
    """
    with psycopg.connect(**SOURCE_DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # batchデータ取得
            cur.execute(
                "SELECT * FROM batch ORDER BY id OFFSET %s LIMIT %s",
                (offset, limit)
            )
            batch_data = cur.fetchall()

            # batch_resultデータ取得
            cur.execute(
                """
                SELECT br.*
                FROM batch_result br
                JOIN batch b ON br.batch_id = b.id
                ORDER BY br.batch_id, br.result_column1
                OFFSET %s LIMIT %s
                """,
                (offset, limit)
            )
            batch_result_data = cur.fetchall()

    return batch_data, batch_result_data


def insert_batches(batch_data, batch_result_data):
    """
    ターゲットデータベースにデータを挿入
    """
    with psycopg.connect(**TARGET_DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # batchを挿入し、新しいIDを取得
            for batch in batch_data:
                cur.execute(
                    "INSERT INTO batch (column1, column2, column3, column4, column5, column6) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    batch[1:]  # IDを除いたデータ
                )
                new_batch_id = cur.fetchone()[0]

                # batch_resultを挿入
                for result in batch_result_data:
                    if result[0] == batch[0]:  # 元のbatch_idが一致するデータ
                        cur.execute(
                            "INSERT INTO batch_result (batch_id, result_column1, result_column2, result_column3) VALUES (%s, %s, %s, %s)",
                            (new_batch_id, result[1], result[2], result[3])
                        )
        conn.commit()


def read_progress():
    """
    進捗ログを読み込み
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return {int(line.strip()) for line in f}
    return set()


def write_progress(offset):
    """
    進捗ログに記録
    """
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{offset}\n")


def process_offset_range(offset, limit, completed_offsets):
    """
    OFFSETとLIMITで指定した範囲を処理
    """
    if offset in completed_offsets:
        print(f"Skipping already processed batch with OFFSET {offset}")
        return

    batch_data, batch_result_data = fetch_batches(offset, limit)
    if batch_data:
        insert_batches(batch_data, batch_result_data)
        write_progress(offset)
    print(f"Processed batch with OFFSET {offset}")


def migrate_data():
    """
    OFFSETとLIMITを使用したデータ移行
    """
    # 進捗ログの読み込み
    completed_offsets = read_progress()

    # ソースデータベースの総データ件数を取得
    with psycopg.connect(**SOURCE_DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM batch")
            total_rows = cur.fetchone()[0]

    # OFFSETとLIMITでデータをバッチ処理
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_offset_range, offset, BATCH_SIZE, completed_offsets)
            for offset in range(0, total_rows, BATCH_SIZE)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    migrate_data()
