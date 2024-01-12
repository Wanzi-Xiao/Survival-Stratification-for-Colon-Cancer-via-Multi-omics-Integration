import pandas as pd

# Load the TSV file
df = pd.read_csv('TCGA-COAD.mirna.tsv', sep='\t')

# Save as CSV
df.to_csv('mirna_original.csv', index=False)
