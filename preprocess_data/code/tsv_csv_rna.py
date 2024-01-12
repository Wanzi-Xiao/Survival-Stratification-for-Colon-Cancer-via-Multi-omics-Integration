import pandas as pd

# Load the TSV file
df = pd.read_csv('TCGA-COAD.htseq_fpkm.tsv', sep='\t')

# Save as CSV
df.to_csv('TCGA-COAD.htseq_fpkm.csv', index=False)
