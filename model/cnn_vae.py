import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

import common as com


class CNN_VAE(nn.Module):
    """Convolutional Variational Autoencoder for spectrogram reconstruction."""

    def __init__(self, device, n_mels, n_frames, z_dim=32, vae=True):
        """
        Initialize CNN VAE model.
        
        Args:
            device (torch.device): Device for model allocation
            n_mels (int): Number of mel frequency bins
            n_frames (int): Number of time frames
            z_dim (int): Latent space dimension
            vae (bool): If True, use VAE; if False, use AE (no KL divergence)
        """
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.flatten_dim = 64 * ((n_mels + 7) // 8) * ((n_frames + 7) // 8)
        self.vae = vae
        
        # Encoder: 3 conv layers with 2x stride downsampling
        self.enc1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.enc2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        if self.vae:
            self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)

        # Decoder: Linear expansion from latent space
        self.dec_fc = nn.Linear(z_dim, self.flatten_dim)
        
        # Decoder: 3 transposed conv layers with 2x stride upsampling
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))

    def encode(self, x):
        """Encode spectrogram to latent space."""
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        self.shape_before_flatten = h.shape[1:]
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        if self.vae:
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation back to spectrogram."""
        h = F.relu(self.dec_fc(z))
        h = torch.unflatten(h, 1, self.shape_before_flatten)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = self.dec3(h)
        return h

    def forward_all(self, x):
        """Forward pass returning all intermediate values."""
        if self.vae:
            mu, logvar = self.encode(x)
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                z = mu
            return self.decode(z), z, mu, logvar
        else:
            mu = self.encode(x)
            return self.decode(mu), mu

    def forward(self, x):
        """Forward pass."""
        return self.forward_all(x)


def VAE_loss_function(recon_x, x, mu, logvar):
    """
    CNN VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        recon_x (torch.Tensor): Reconstructed spectrogram
        x (torch.Tensor): Original spectrogram
        mu (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
    
    Returns:
        tuple: (reconstruction_loss, kl_divergence_loss)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld_loss


def AE_loss_function(recon_x, x):
    """
    CNN Autoencoder loss function: weighted combination of MSE and SSIM loss.
    
    Args:
        recon_x (torch.Tensor): Reconstructed spectrogram
        x (torch.Tensor): Original spectrogram
    
    Returns:
        torch.Tensor: Weighted reconstruction loss
    """
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    ssim_loss = 1 - ssim(recon_x, x, data_range=6.0)
    recon_loss = 0.5 * mse_loss + 0.5 * ssim_loss
    return recon_loss
