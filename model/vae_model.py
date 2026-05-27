import torch
import torch.nn as nn
import torch.nn.functional as F
import common as com


class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection.
    
    Based on https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    def __init__(self, device, x_dim=640, h_dim=128, z_dim=8, vae=True):
        """
        Initialize VAE model.
        
        Args:
            device (torch.device): Device for model allocation
            x_dim (int): Input feature dimension
            h_dim (int): Hidden layer dimension
            z_dim (int): Latent space dimension
            vae (bool): If True, use VAE; if False, use AE (no KL divergence)
        """
        super().__init__()
        self.x_dim = x_dim
        self.device = device
        self.vae = vae

        self.enc1 = nn.Linear(x_dim, h_dim)
        self.enc2 = nn.Linear(h_dim, h_dim)
        self.enc3 = nn.Linear(h_dim, h_dim)

        self.fc_mu = nn.Linear(h_dim, z_dim)
        if self.vae:
            self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        self.dec1 = nn.Linear(z_dim, h_dim)
        self.dec2 = nn.Linear(h_dim, h_dim)
        self.dec3 = nn.Linear(h_dim, h_dim)
        self.dec4 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
        """Encode input to latent space."""
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        mu = self.fc_mu(h)
        if self.vae:
            return mu, self.fc_logvar(h)
        else:
            return mu

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation back to input space."""
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        return self.dec4(h)

    def forward_all(self, x):
        """Forward pass returning all intermediate values."""
        if self.vae:
            mu, logvar = self.encode(x.view(-1, self.x_dim))
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                z = mu
            return self.decode(z), z, mu, logvar
        else:
            mu = self.encode(x.view(-1, self.x_dim))
            return self.decode(mu), mu

    def forward(self, x):
        """Forward pass."""
        return self.forward_all(x)


def VAE_loss_function(recon_x, x, mu, logvar, x_dim=640):
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        recon_x (torch.Tensor): Reconstructed output
        x (torch.Tensor): Original input
        mu (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
        x_dim (int): Input dimension
    
    Returns:
        tuple: (reconstruction_loss, kl_divergence_loss)
    """
    reconst_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconst_loss, kld


def AE_loss_function(recon_x, x, x_dim=640):
    """
    Autoencoder loss function: reconstruction loss only.
    
    Args:
        recon_x (torch.Tensor): Reconstructed output
        x (torch.Tensor): Original input
        x_dim (int): Input dimension
    
    Returns:
        torch.Tensor: Reconstruction loss
    """
    reconst_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='mean')
    return reconst_loss
