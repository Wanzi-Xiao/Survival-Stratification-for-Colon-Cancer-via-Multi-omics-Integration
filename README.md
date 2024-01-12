# Survival Stratification for Colon Cancer via Multi-omics Integration

## Overview
Our research project integrates two omics in the colon cancer: gene expression(RNA-seq) and miRNA. We use three models to extract features from data to obtain a joint latent space: Variational Autoencoder(VAE), PCA and UMAP through 10-fold cross-validation. We group the patients in an unsupervised manner though k-means clustering and we build an SVM classfier for predicting the labels of validation groups. The results show that VAE model has slightly better performance than the other two.

## Contributors
- Wanzi Xiao
- Xueke Jin

## Data Resource
### Table 1: Overview information of the colon-cancer dataset

| Dataset Info       |                                        |
| ------------------ | -------------------------------------- |
| **Source**         | UCSC Xena data portal, TCGA            |
| **Additional label** | Vital_status, days_to_birth, days_to_death, stage |
| **Omics type**     | Gene expression RNA-seq |
| **Feature number** | 54186 |
| **Sample number**  | 513 |
| **Omics type**     | miRNA expression |
| **Feature number** | 1881 |
| **Sample number**  | 462 |

## Methodology
1. **Data Preprocessing**: Normalization and filtering of RNA-seq and miRNA data.
### Figure 1: [Data preprocess workflow]
![Data preprocess workflow](https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/Data%20preprocess%20workflow.png)
3. **Feature Selection**: Using Cox Proportional Hazards Model.
4. **Clustering & Classification**: K-means and SVM.
5. **Model Evaluation**: Concordance Index, ROC Curve, AUC.

## Results
Includes hazard ratio, Kaplan-Meier curves, risk score visualizations, and correlation matrix.

## References
[List of references]

## Contact
- Wanzi Xiao: <wanzix@andrew.cmu.edu>
- Xueke Jin: <xuekej@andrew.cmu.edu>
