import csv
import random
from datetime import datetime, timedelta
import psycopg

# データベース接続情報
TARGET_DB_CONFIG = {
    'dbname': 'source_db',
    'user': 'test_user',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

# 都道府県リスト
prefectures = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県",
    "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県",
    "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県",
    "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

# 性別リスト
genders = ["男性", "女性"]

# データ生成
num_records = 100000

# データを挿入
def insert_data():
    with psycopg.connect(**TARGET_DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for i in range(1, num_records + 1):
                name = f"名前_{i}"
                prefecture = random.choice(prefectures)
                annual_income = random.randint(1000, 20000) * 1000  # 年収（20万円〜2000万円）
                gender = random.choice(genders)
                is_active = random.choice([0, 1])  # 存在フラグ（0または1）
                registration_date = datetime.now() - timedelta(days=random.randint(0, 365 * 10))  # 過去10年間の日付

                cur.execute(
                    """
                    INSERT INTO large_dataset (name, prefecture, annual_income, gender, is_active, registration_date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (name, prefecture, annual_income, gender, bool(is_active), registration_date.strftime("%Y-%m-%d"))
                )
            conn.commit()

if __name__ == "__main__":
    for i in range(10):
        insert_data()
    print(f"{num_records}件のデータをデータベースにインポートしました。")
