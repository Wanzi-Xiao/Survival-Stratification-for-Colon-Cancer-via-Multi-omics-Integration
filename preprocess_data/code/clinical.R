if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")
library(TCGAbiolinks)

project = "TCGA-COAD"
query <- GDCquery(project = project,
                 data.category = "Clinical",
                 file.type = "xml")
GDCdownload(query)
clinical <- GDCprepare_clinic(query, clinical.info = "patient")

write.csv(clinical, "COAD_Clinical_Data.csv", row.names = FALSE)


