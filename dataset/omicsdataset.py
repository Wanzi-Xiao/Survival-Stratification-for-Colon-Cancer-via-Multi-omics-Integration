from torch.utils.data import Dataset
import numpy as np
import torch

class OmicsDataset(Dataset):
    def __init__(self, omics_1, omics_2):
        self.omics_1 = omics_1
        self.omics_2 = omics_2
        self.sample_ids = omics_1.index

    def __len__(self):
        return len(self.omics_1)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        omics_1_sample = self.omics_1.iloc[idx].values.astype(np.float32)
        omics_2_sample = self.omics_2.iloc[idx].values.astype(np.float32)
        return sample_id, omics_1_sample, omics_2_sample
