import psycopg
import random
from faker import Faker

# PostgreSQL接続情報
DB_CONFIG = {
    'dbname': 'source_db',
    'user': 'test_user',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

# データ生成と挿入
def insert_data():
    # Fakerインスタンス
    fake = Faker()

    # PostgreSQLに接続
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # 1. batchテーブルに100万件挿入
            print("Inserting into batch...")
            batch_data = [
                (fake.word(), fake.word(), fake.word(), fake.word(), fake.word(), fake.word())
                for _ in range(500000)
            ]
            cur.executemany(
                "INSERT INTO batch (column1, column2, column3, column4, column5, column6) VALUES (%s, %s, %s, %s, %s, %s)",
                batch_data
            )
            conn.commit()
            print("Batch data inserted.")
            cur.execute("select max(id),min(id) from batch")
            data = cur.fetchall()[0]
            max = data[0]
            min = data[1]

            # 2. batch_resultテーブルに挿入
            print("Inserting into batch_result...")

            batch_result_data = []

            for _ in range(1000000):
                batch_result_data.append((
                    random.randint(min,max),
                    f"company_id - {random.randint(0,5000000)} - {random.randint(5000000,25000000)}",
                    random.randint(5,60000000),
                    random.randint(60000000,100000000)
                ))

            cur.executemany(
                "INSERT INTO batch_result (batch_id, result_column1, result_column2, result_column3) VALUES (%s, %s, %s, %s)",
                batch_result_data
            )
            conn.commit()
            print("Batch_result data inserted.")

# 実行
if __name__ == "__main__":
    insert_data()
