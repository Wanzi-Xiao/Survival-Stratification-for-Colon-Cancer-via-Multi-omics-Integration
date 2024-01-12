from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

class OmicsPCA:
    def __init__(self, n_components=2):
        """
        Initialize the OmicsPCA class with the number of principal components.

        :param n_components: The number of principal components to keep.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.principal_components = None

    def fit(self, data):
        """
        Fit PCA on the integration of two omics datasets and transform the data.

        :data: DataFrame or numpy array for the combined omics dataset.
        :return: Transformed data with reduced dimensions as a numpy array.
        """
        # Ensure that the input data are DataFrames
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        index = data.index
        transformed_data = self.pca.fit_transform(data)
        
        # Fit PCA on the combined dataset and transform the data
        # self.principal_components = self.pca.fit_transform(data)
        self.principal_components = pd.DataFrame(
            transformed_data,
            index=index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        return self.principal_components
    
    def transform(self, data):
        """
        Transform the data using the already fitted PCA model.

        :data: DataFrame or numpy array to be transformed.
        :return: Transformed data with reduced dimensions as a numpy array.
        """
        # Ensure that the input data are DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        transformed_data = self.pca.transform(data)
        index = data.index

        return pd.DataFrame(
            transformed_data,
            index=index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )

    def explained_variance_ratio(self):
        """
        Return the amount of variance explained by each of the selected components.

        :return: Array of variance explained by each principal component.
        """
        
        return self.pca.explained_variance_ratio_

    def get_components(self):
        """
        Return the principal components as a DataFrame with appropriate naming.

        :return: DataFrame of the principal components.
        """
        if self.principal_components is None:
            raise ValueError("You must fit the model before getting components.")
        
        columns = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(self.principal_components, columns=columns)

    def save_latent_space(self, file_path):
        """
        Save the reduced data (latent space) to a CSV file.

        :param file_path: The path to the file where the latent space will be saved.
        """
        if self.principal_components is None:
            raise ValueError("The PCA model must be fit before saving the latent space.")
        
        # Convert the numpy array of principal components to a DataFrame
        # principal_components_df = pd.DataFrame(
        #     reduced_data,
        #     columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])]
        # )
        
        # Save the DataFrame to a CSV file
        # principal_components_df.to_csv(file_path)
        self.principal_components.to_csv(file_path)

    def plot_pca_space(self, labels=None, file_path=None):
        """
        Plot the PCA space using the first two principal components.

        :param labels: Optional. Labels for each data point for coloring.
        :param file_path: Optional. Path to save the plot. If None, plot will be shown.
        """
        if self.principal_components is None:
            raise ValueError("The PCA model must be fit before plotting the PCA space.")
        
        # Check if the PCA has at least 2 components
        if self.principal_components.shape[1] < 2:
            raise ValueError("There must be at least 2 principal components to plot the PCA space.")

        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(self.principal_components.iloc[:, 0], self.principal_components.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
        else:
            plt.scatter(self.principal_components.iloc[:, 0], self.principal_components.iloc[:, 1], alpha=0.5)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Space')

        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()
