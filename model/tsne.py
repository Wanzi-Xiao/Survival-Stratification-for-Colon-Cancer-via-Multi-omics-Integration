from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

class OmicsTSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        """
        Initialize the OmicsTSNE class with t-SNE parameters.

        :param n_components: The number of dimensions to reduce the data to.
        :param perplexity: The perplexity parameter for t-SNE.
        :param learning_rate: The learning rate for t-SNE.
        :param n_iter: The number of iterations for optimization.
        """
        self.n_components = n_components
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                         learning_rate=learning_rate, n_iter=n_iter)
        self.transformed_data = None

    def fit_transform(self, data):
        """
        Fit t-SNE on the dataset and transform the data.

        :param data: DataFrame or numpy array for the omics dataset.
        :return: Transformed data as a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Fit and transform the data using t-SNE
        transformed_data = self.tsne.fit_transform(data)
        self.transformed_data = pd.DataFrame(transformed_data, index=data.index, 
                               columns=[f'TSNE{i+1}' for i in range(self.n_components)])
        return self.transformed_data

    def save_latent_space(self, file_path):
        """
        Save the transformed data (latent space) to a CSV file.

        :param file_path: The path to the file where the latent space will be saved.
        """
        if self.transformed_data is None:
            raise ValueError("The t-SNE model must be fit before saving the latent space.")
        self.transformed_data.to_csv(file_path)

    def plot_tsne_space(self, labels=None, file_path=None):
        """
        Plot the t-SNE space using the transformed components.

        :param labels: Optional. Labels for each data point for coloring.
        :param file_path: Optional. Path to save the plot. If None, plot will be shown.
        """
        if self.transformed_data is None:
            raise ValueError("The t-SNE model must be fit before plotting the t-SNE space.")

        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(self.transformed_data.iloc[:, 0], self.transformed_data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
        else:
            plt.scatter(self.transformed_data.iloc[:, 0], self.transformed_data.iloc[:, 1], alpha=0.5)

        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.title('t-SNE Space')

        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()
