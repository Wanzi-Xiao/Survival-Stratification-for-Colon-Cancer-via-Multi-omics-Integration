# Survival Stratification for Colon Cancer via Multi-omics Integration

## Overview
Our research project integrates two omics in the colon cancer: gene expression(RNA-seq) and miRNA. We use three models to extract features from data to obtain a joint latent space: Variational Autoencoder(VAE), PCA and UMAP through 10-fold cross-validation. We group the patients in an unsupervised manner though k-means clustering and we build an SVM classfier for predicting the labels of validation groups. The results show that VAE model has slightly better performance than the other two.

## Contributors
- Wanzi Xiao: <wanzix@andrew.cmu.edu>
- Xueke Jin: <xuekej@andrew.cmu.edu>

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
<p align="center">
<img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/Data%20preprocess%20workflow.png" width="1000" height="150">
  </p>

<p align="center">
  Figure 1: Data preprocess workflow
  </p>

2. **Dimension Reduction**: Using VAE, PCA and UMAP models.
<p align="center">
  <img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/Method%20workflow.png" width="270" height="400">
  </p>

<p align="center">
  Figure 2: Method workflow
  </p>

<p align="center">
  <img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/Variational%20Autoencoder%20Architecture.png" width="500" height="400">
  </p>

<p align="center">
  Figure 3: Variational Autoencoder Architecture
  </p>
  
3. **Feature Selection**: Using Cox Proportional Hazards Model.
4. **Clustering & Classification**: K-means and SVM.
5. **Model Evaluation**: Concordance Index, ROC Curve, AUC.

## Results
Includes hazard ratio, Kaplan-Meier curves, risk score visualizations, and correlation matrix. In Figure 4, kest_k means the optimal number of clusters determined when using the k-clustering method.

<p align="center">
  <img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/KM%20Curve.png" width="1000" height="370">
  </p>

<p align="center">
  Figure 4: Kaplan-Meier Curve comparison of three models 
  </p>

<p align="center">
  <img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/Risk%20Score.png" width="1000" height="370">
  </p>

<p align="center">
  Figure 5: Risk score comparison of three models 
  </p>

<p align="center">
  <img src="https://github.com/Wanzi-Xiao/Survival-Stratification-for-Colon-Cancer-via-Multi-omics-Integration/blob/main/figures/ROC.png" width="1000" height="370">
  </p>

<p align="center">
  Figure 6: ROC curve comparison of three models 
  </p>

## Conclusion
In summary, the VAE model consistently provided the most detailed insights across various analyses, making it a robust choice for nuanced data stratification and risk assessment. PCA proved to be reliable and interpretable, while UMAP’s performance suggests it may benefit from further optimization. The choice of model should align with the analytical goals and clinical context of the study.
Overall, the integration of various types of omics data demonstrates a powerful approach to under- standing cancer.In the future study,the methodologies applied here can be used to stratify patients in clinical trials, ensuring that trials are more targeted and potentially more effective.
