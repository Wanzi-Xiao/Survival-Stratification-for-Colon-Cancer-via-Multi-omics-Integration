import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from .loss import vae_loss
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from .earlystop import EarlyStopping


class NetManager:
    """
    Net manager for the VAE
    """

    def __init__(
            self,
            model,
            device,
            train_loader=None,
            test_loader=None,
            lr=1e-3):
        """
        Constructor
        """
        self.model = model.to(device)
        self.device = device
        self.writer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def set_writer(self, board_name):
        """
        Sets a torch writer object. The logs will be generated in logs/name
        """
        if isinstance(self.writer, SummaryWriter):
            self.writer.close()

        if board_name is None:
            self.writer = None
        else:
            self.writer = SummaryWriter("logs/" + board_name)

    def train(self, epochs, log_interval=10):
        self.model.train()
        for epoch in range(epochs):
            early_stopping = EarlyStopping(patience=10, min_delta=0.1)
            train_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                sample_ids, omics_1, omics_2 = batch

                omics_1 = omics_1.to(self.device)
                omics_2 = omics_2.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(omics_1, omics_2)

                # Calculate loss
                loss = vae_loss(recon_batch, omics_1, omics_2, mu, logvar, loss_method='MSE')
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                val_loss = self.validate()

                if self.writer is not None:
                    self.writer.add_scalar('Loss/val', val_loss, epoch)

                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

                if batch_idx % log_interval == 0 and self.writer is not None:
                    current_loss = train_loss / log_interval
                    self.writer.add_scalar('Loss/train', current_loss, epoch * len(self.train_loader) + batch_idx)

            average_loss = train_loss / len(self.train_loader)
            print(f'Epoch {epoch}, Train Loss: {average_loss}, Val Loss: {val_loss}')
            # print(f'Epoch {epoch}, Loss: {average_loss}')

            if self.writer is not None:
                self.writer.flush()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                sample_ids, omics_1, omics_2 = batch

                omics_1 = omics_1.to(self.device)
                omics_2 = omics_2.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(omics_1, omics_2)

                # Calculate loss
                loss = vae_loss(recon_batch, omics_1, omics_2, mu, logvar, loss_method='MSE')
                val_loss += loss.item()
        return val_loss / len(self.test_loader)

    def save_net(self, file_path):
        """
        Save the model state.

        Parameters:
            file_path (str): The path to the file where the state dict will be saved.
        """
        torch.save(self.model.state_dict(), file_path)
    
    def plot_latent_space(self, data_loader, file_path=None):
        self.model.eval()
        latent_vectors = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                sample_ids, omics_1, omics_2 = batch
                
                omics_1 = omics_1.to(self.device)
                omics_2 = omics_2.to(self.device)
                
                # Encode the input data to get the latent space variables
                mu, _ = self.model.encode(omics_1, omics_2)
                latent_vectors.append(mu.to('cpu').numpy())

        # Concatenate all latent vectors and apply t-SNE for visualization
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(latent_vectors)

        # Plot the t-SNE transformed latent vectors
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        plt.title('Latent Space Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # Save the plot to a file or display it, depending on 'file_path'
        if file_path is not None:
            plt.savefig(file_path)
            print(f"Latent space plot saved to {file_path}")
            plt.close()
        else:
            NotImplementedError
