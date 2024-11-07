# 入力ファイル名と出力ファイル名を定義
input_file = 'python-code/lab/code.csv'  # 改行区切りのデータが含まれるファイル
output_file = 'python-code/lab/new_code.csv'  # カンマ区切りのデータを書き込むファイル

# 入力ファイルを読み込み、改行で区切られたデータを取得
with open(input_file, 'r', encoding='UTF-8') as infile:
    lines = infile.readlines()

# データをカンマ区切りの形式に変換
# すべての行を1行にまとめ、その間にカンマを挿入
comma_separated_data = ','.join(line.strip() for line in lines)

# 出力ファイルに書き込み
with open(output_file, 'w', encoding='UTF-8') as outfile:
    outfile.write(comma_separated_data)

print(f"変換が完了しました。結果は {output_file} に書き込まれました。")

