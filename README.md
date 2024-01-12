# Survival Stratification for Colon Cancer via Multi-omics Integration

## Overview
This project develops a computational method for survival stratification in colon adenocarcinoma using multi-omics data.

## Contributors
- Wanzi Xiao
- Xueke Jin

## Data Resource
### Table 1: Overview information of the colon-cancer dataset

| Dataset Info | Colon Cancer |
| --------------- | --------------- |
| Source      | UCSC Xena data portal, TCGA  |
| Additional label     | Vital_status, days_to_birth, days_to_death, stage    |
| Omics type | Gene expression RNA-seq | miRNA expression |
| Feature number      | 54186         | 1881        |
| Sample number       | 513         | 462        |

## Methodology
1. **Data Preprocessing**: Normalization and filtering of RNA-seq and miRNA data.
2. **Feature Selection**: Using Cox Proportional Hazards Model.
3. **Clustering & Classification**: K-means and SVM.
4. **Model Evaluation**: Concordance Index, ROC Curve, AUC.

## Results
Includes hazard ratio, Kaplan-Meier curves, risk score visualizations, and correlation matrix.

## References
[List of references]

## Contact
- Wanzi Xiao: <wanzix@andrew.cmu.edu>
- Xueke Jin: <xuekej@andrew.cmu.edu>
