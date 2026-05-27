import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

import common as com


class CNN_VAE(nn.Module):
    """Convolutional VAE with hierarchical classification head."""

    def __init__(self, device, n_mels, n_frames, z_dim=32, n_classes=3, n_sub=3, vae=True):
        """
        Initialize CNN VAE with classification head.
        
        Args:
            device (torch.device): Device for model allocation
            n_mels (int): Number of mel frequency bins
            n_frames (int): Number of time frames
            z_dim (int): Latent space dimension
            n_classes (int): Number of machine classes
            n_sub (int): Number of sub-classes (sections) per machine
            vae (bool): If True, use VAE; if False, use AE (no KL divergence)
        """
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.flatten_dim = 64 * n_mels // 8 * n_frames // 8
        self.vae = vae
        self.n_classes = n_classes
        self.n_sub = n_sub
        
        # Encoder: 3 conv layers with 2x stride downsampling
        self.enc1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.enc2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        if self.vae:
            self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)

        # Classification head: hierarchical machine type and section prediction
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.ReLU(),
            nn.Linear(z_dim * 2, n_classes * n_sub)
        )

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

    def forward(self, x):
        """
        Forward pass with reconstruction and classification outputs.
        
        Args:
            x (torch.Tensor): Input spectrogram (batch, 1, n_mels, n_frames)
        
        Returns:
            tuple: Depends on mode (VAE or AE):
                - VAE: (reconstruction, z, mu, logvar, class_probs)
                - AE: (reconstruction, mu, class_probs)
        """
        if self.vae:
            mu, logvar = self.encode(x)
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                z = mu
            logits = self.classifier(z)
        else:
            mu = self.encode(x)
            logits = self.classifier(mu)

        class_prob = torch.sigmoid(logits).view(-1, self.n_classes, self.n_sub)
        
        if self.vae:
            return self.decode(z), z, mu, logvar, class_prob
        else:
            return self.decode(mu), mu, class_prob


def VAE_loss_function(recon_x, x, mu, logvar, pred_probs, target_class):
    """
    CNN VAE loss function: reconstruction + KL divergence + classification loss.
    
    Args:
        recon_x (torch.Tensor): Reconstructed spectrogram
        x (torch.Tensor): Original spectrogram
        mu (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
        pred_probs (torch.Tensor): Predicted class probabilities
        target_class (torch.Tensor): Target class labels
    
    Returns:
        tuple: (reconstruction_loss, kl_divergence_loss, classification_loss)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.binary_cross_entropy(pred_probs, target_class, reduction='mean')
    return recon_loss, kld_loss, class_loss


def AE_loss_function(recon_x, x, pred_probs, target_class):
    """
    CNN Autoencoder loss function: reconstruction loss + classification loss.
    
    Args:
        recon_x (torch.Tensor): Reconstructed spectrogram
        x (torch.Tensor): Original spectrogram
        pred_probs (torch.Tensor): Predicted class probabilities
        target_class (torch.Tensor): Target class labels
    
    Returns:
        tuple: (reconstruction_loss, classification_loss)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    class_loss = F.binary_cross_entropy(pred_probs, target_class, reduction='mean')
    return recon_loss, class_loss
