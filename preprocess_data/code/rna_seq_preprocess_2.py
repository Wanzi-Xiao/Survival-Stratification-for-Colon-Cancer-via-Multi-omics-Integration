import pandas as pd
import os

data_dir = ""
rna_seq_path = "TCGA-COAD.htseq_fpkm.csv"
rna_seq_filtered_y_chromosome_path =  "rna_seq_filtered_y_chromosome.csv"
path_to_save = "rna-seq.csv"

original_rna_data = pd.read_csv(os.path.join(data_dir, rna_seq_path))

# print out the length of the original rna-seq data file
print(f"The dimension of original RNA-seq data is {len(original_rna_data)}.")

#filter out genes that target Y chromosome
filtered_genes = pd.read_csv(os.path.join(data_dir, rna_seq_filtered_y_chromosome_path))
filtered_gene_list = filtered_genes['x'].tolist()
pattern = '|'.join([x + r"\.\d+" for x in filtered_gene_list])
filtered_original_data = original_rna_data[original_rna_data['Ensembl_ID'].str.contains(pattern)]

# filter genes that have all zeros for all samples
columns_to_check = filtered_original_data.columns.difference(['Ensembl_ID'])
filtered_genes = filtered_original_data.replace(0, pd.NA).dropna(subset=columns_to_check, how='all')

print(f"The dimension of filtered RNA-seq data is {len(filtered_genes)}.")

# Normalisation
filtered_genes = filtered_genes[columns_to_check].fillna(0)
clean_filtered_genes = filtered_genes.copy()
clean_filtered_genes[columns_to_check] = (
    clean_filtered_genes[columns_to_check] - clean_filtered_genes[columns_to_check].min()) / (
    clean_filtered_genes[columns_to_check].max() - clean_filtered_genes[columns_to_check].min())

clean_filtered_genes = clean_filtered_genes.join(filtered_original_data["Ensembl_ID"])
cols = list(clean_filtered_genes.columns)
cols = [cols[-1]] + cols[:-1]
clean_filtered_genes = clean_filtered_genes[cols]

clean_filtered_genes.to_csv(os.path.join(data_dir, path_to_save), index=False)

