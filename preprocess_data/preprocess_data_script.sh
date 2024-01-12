#!/bin/bash

# Download the file
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.mirna.tsv.gz
# Downloading the RNA-seq data
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.htseq_fpkm.tsv.gz
# Unzip the downloaded file
gunzip TCGA-COAD.mirna.tsv.gz
# Unzipping the downloaded file
gunzip TCGA-COAD.htseq_fpkm.tsv.gz 
# Execute the Python scripts from the 'code' directory
python code/tsv_csv_mirna.py
python code/normalize.py

# Running Python script for initial processing
python code/tsv_csv_rna.py

# Running R script for further processing
Rscript code/rna_seq_preprocess_1.R

# Running second Python script for final processing
python code/rna_seq_preprocess_2.py
# allign two data based on same sample ids
python code/common.py
# tranpose data 
python code/transpose.py

