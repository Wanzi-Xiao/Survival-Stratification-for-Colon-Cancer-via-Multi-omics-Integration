import pandas as pd


df = pd.read_csv('filtered_mirna.csv')
df1 = pd.read_csv('filtered_rna-seq.csv')

df_transposed = df.T  
df1_transposed = df1.T

df_transposed.to_csv('mirna.csv', header=False)
df1_transposed.to_csv('rna.csv', header=False)
