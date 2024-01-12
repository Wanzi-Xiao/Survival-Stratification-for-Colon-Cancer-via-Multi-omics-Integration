import torch
from torch.nn import functional as F

def vae_loss(recon_x, dna_methylation, gene_expression, mu, logvar, E = None, tri_matrix = None, loss_method='MSE'):
    """
    Compute the loss for a Variational Autoencoder with selectable loss methods.

    Parameters:
        recon_x (Tensor): The reconstructed output from the VAE decoder.
        dna_methylation (Tensor): The original DNA methylation data.
        gene_expression (Tensor): The original gene expression data.
        mu (Tensor): The mean from the VAE encoder's latent space.
        logvar (Tensor): The log variance from the VAE encoder's latent space.
        loss_method (str): The method for calculating the loss ('MSE' or 'MTLR').

    Returns:
        Tensor: The total loss.
    """
    if loss_method == 'MSE':
        # Mean Squared Error Loss
        mse_dna = F.mse_loss(recon_x[:, :dna_methylation.size(1)], dna_methylation, reduction='sum')
        mse_gene = F.mse_loss(recon_x[:, dna_methylation.size(1):], gene_expression, reduction='sum')
        mse = mse_dna + mse_gene

        # Kullback-Leibler Divergence Loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = mse + kld

    elif loss_method == 'MTLR':
        # Implement the MTLR loss calculation here
        total_loss = None  # Replace None with the actual MTLR loss computation

    else:
        raise ValueError("Invalid loss method specified. Choose 'MSE' or 'MTLR'.")

    return total_loss

def MTLR_survival_loss(y_pred, y_true, E, tri_matrix, reduction='mean'):
    """
    Compute the MTLR survival loss
    """
    # Get censored index and uncensored index
    censor_idx = []
    uncensor_idx = []
    for i in range(len(E)):
        # If this is a uncensored data point
        if E[i] == 1:
            # Add to uncensored index list
            uncensor_idx.append(i)
        else:
            # Add to censored index list
            censor_idx.append(i)

    # Separate y_true and y_pred
    y_pred_censor = y_pred[censor_idx]
    y_true_censor = y_true[censor_idx]
    y_pred_uncensor = y_pred[uncensor_idx]
    y_true_uncensor = y_true[uncensor_idx]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = - (torch.sum(torch.log(reduc_phi_censor)) + torch.sum(torch.log(reduc_phi_uncensor)) - torch.sum(torch.log(reduc_z_censor)) - torch.sum(torch.log(reduc_z_uncensor)))

    if reduction == 'mean':
        loss = loss / E.shape[0]

    return loss


