import pandas as pd

material_name = 'NOX'

coordinate_path = f'python-code/lab/out_data/coordinate/csv/OX_NMHC_TEMP_coordinate.csv'
CH4_path = f'python-code/lab/out_data/coordinate/csv/{material_name}.csv'

a = pd.read_csv(coordinate_path, header = 0, index_col = None)
b = pd.read_csv(CH4_path, header = 0, index_col = None)

a_d = a['国環研局番']
b = b['測定局コード']

a_d = list(a_d)
b = list(set(b))

a_d.sort()
b.sort()

a_b_and = set(a_d) & set(b)
li = list(a_b_and)
li = list(set(li))

li.sort()

c = []
d = a.query('国環研局番 == @li')
d.to_csv(f'python-code/lab/out_data/coordinate/csv/{material_name}_OX_NMHC_TEMP_coordinate.csv', header = d.columns, index = False)