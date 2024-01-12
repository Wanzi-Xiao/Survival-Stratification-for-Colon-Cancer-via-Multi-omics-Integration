import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your data
df = pd.read_csv('miRNA_original.csv', index_col=0)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to each row/feature
# This is done by applying the scaler along the columns axis (axis=1)
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Optionally, you can save the normalized data back to a CSV
df_normalized.to_csv('miRNA_normalized.csv')

