import torch
from torch import nn
from torch.nn import functional as F

class BaseVAE(nn.Module):
    """
    Base abstract class for Variational Autoencoders.
    """
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, x):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class EmbedVAE(BaseVAE):
    """
    Embed Variational Autoencoder based on the provided architecture,
    customized for gene expression and DNA methylation data.
    """
    def __init__(self, omics_1_dim, omics_2_dim, z_dim=128):
        super(EmbedVAE, self).__init__()
        total_input_dim = omics_1_dim + omics_2_dim
        # Linear => BatchNorm => Dropout => LeakyReLu

        # Separate encoder pathways for miRNA and gene expression 
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(omics_1_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(omics_2_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Combine the outputs of the separate pathways
        self.encoder_combined = nn.Sequential(
            nn.Linear(4096 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        
        # Decoder will generate outputs for both omics together
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, total_input_dim),  
            nn.BatchNorm1d(total_input_dim),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()  # Use Sigmoid if the output needs to be normalized between 0 and 1
        )

    def encode(self, omics_1_dim, omics_2_dim):
        h1_omics_1 = self.encoder_omics_1(omics_1_dim)
        h1_omics_2 = self.encoder_omics_2(omics_2_dim)
        h_combined = torch.cat((h1_omics_1, h1_omics_2), dim=1)
        h2 = self.encoder_combined(h_combined)
        # check if encoder has NaN values
        assert not torch.isnan(h2).any(), "NaN values after encoder"
        return self.fc_mu(h2), self.fc_logvar(h2)

    def decode(self, z):
        # check if decoder has NaN values
        assert not torch.isnan(self.decoder(z)).any(),  "NaN values after decoder"
        return self.decoder(z)

    def forward(self, omics_1_dim, omics_2_dim):
        mu, logvar = self.encode(omics_1_dim, omics_2_dim)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
