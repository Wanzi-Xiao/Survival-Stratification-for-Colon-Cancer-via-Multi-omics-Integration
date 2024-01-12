import pandas as pd
from sklearn.model_selection import train_test_split
from umap import UMAP
import matplotlib.pyplot as plt

class UMAPReducer:
    def __init__(self, n_components=128, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = UMAP(n_components=n_components, random_state=random_state)

    def fit_transform(self, omics_1_data, omics_2_data, validation_split=0.4):
        combined_data = pd.concat([omics_1_data, omics_2_data], axis=1)
        train_data, validation_data = train_test_split(combined_data, test_size=validation_split)
        self.model.fit(train_data)
        umap_train_data = self.model.transform(train_data)
        umap_validation_data = self.model.transform(validation_data)
        return umap_train_data, umap_validation_data
    
    def plot_umap(self, data, title, file_path, labels):
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Spectral', alpha=0.5)
        plt.title(title)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.colorbar()
        plt.savefig(file_path)
        plt.close()
