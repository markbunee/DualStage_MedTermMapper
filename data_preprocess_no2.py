import pandas as pd

# 读取Excel文件（指定列）
df = pd.read_excel(r'F:\final\origin_dataset\data_test.xlsx', usecols=[2, 3], engine='openpyxl')  # 索引从0开始，2是第三列，3是第四列

# 生成CSV文件
df.to_csv('./map2.csv', index=False, encoding='utf-8-sig')