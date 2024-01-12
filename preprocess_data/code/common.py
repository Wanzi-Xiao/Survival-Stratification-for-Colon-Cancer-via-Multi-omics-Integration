import pandas as pd

# 载入数据
file1 = 'rna-seq.csv'
file2 = 'miRNA_normalized.csv'

df1 = pd.read_csv(file1, index_col=0)
df2 = pd.read_csv(file2, index_col=0)

# 找出共有的样本 ID
common_ids = df1.columns.intersection(df2.columns)

# 过滤数据
filtered_df1 = df1[common_ids]
filtered_df2 = df2[common_ids]

# 保存到新文件
filtered_df1.to_csv('filtered_rna-seq.csv')
filtered_df2.to_csv('filtered_mirna.csv')

