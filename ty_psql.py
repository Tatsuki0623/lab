import tt_psql
import psycopg
from psycopg.rows import dict_row
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

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

def main():
    batches = tt_psql.fetch_batches(0, 10000)
    batches_li = [batches[i:i+1000] for i in range(0, 10000, 1000)]

    s =time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        results = [ex.submit(tt_psql.insert_batch, batch) for batch in batches_li]
        for result in as_completed(results):
            if result.result():
                print("成功")
            else:
                print('失敗')
        print("exit")
                    

    e = time.time()

    print(f"Process: {e-s}秒")   

if __name__ == "__main__":
    main()
