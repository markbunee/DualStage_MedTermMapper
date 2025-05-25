import pandas as pd

# 读取CSV文件
input_file = r'F:\final\map2.csv'  # 输入文件名
output_file = r'F:\final\map3.csv'  # 输出文件名

# 读取数据到DataFrame
df = pd.read_csv(input_file)

# 打印原始数据
print("原始数据：")
print(df)

# 定义要交换的两列索引或名称
col1, col2 = 0, 1  # 假设交换第1列和第2列（索引从0开始）

# 交换两列的位置
cols = list(df.columns)
cols[col1], cols[col2] = cols[col2], cols[col1]
df = df[cols]

# 打印交换后的数据
print("\n交换后的数据：")
print(df)

# 保存结果到新的CSV文件
df.to_csv(output_file, index=False)
print(f"\n结果已保存到 {output_file}")