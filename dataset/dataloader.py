import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .omicsdataset import OmicsDataset

def create_omics_dataloaders(omics_1_path, omics_2_path, indices, batch_size=32, validation_split=0.4):
    omics_1 = pd.read_csv(omics_1_path, index_col=0)
    omics_2 = pd.read_csv(omics_2_path, index_col=0)

    # Subset the dataset based on provided indices
    dataset = OmicsDataset(omics_1.iloc[indices], omics_2.iloc[indices])

    # Split the subset into internal training and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader

def export_latent_space_to_csv(model, data_loader, file_path, device):
    model.eval()  # Set the model to evaluation mode
    latent_space_representations = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in data_loader:
            sample_ids, omics_1, omics_2 = batch
            omics_1 = omics_1.to(device)
            omics_2 = omics_2.to(device)
            mu, _ = model.encode(omics_1, omics_2)
            latent_space_representations.extend(mu.to('cpu').numpy().tolist())
            all_sample_ids.extend(sample_ids)

    # Create a DataFrame with the latent space representations and sample IDs as the index
    df = pd.DataFrame(latent_space_representations, index=all_sample_ids)
    df.to_csv(file_path)