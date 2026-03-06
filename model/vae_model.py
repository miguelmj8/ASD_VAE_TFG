import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Mostly borrowed from PyTorch example.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    def __init__(self, device, x_dim=640, h_dim=128, z_dim=8):
        super().__init__()
        self.x_dim = x_dim
        self.device = device

        self.enc1 = nn.Linear(x_dim, h_dim)
        self.enc2 = nn.Linear(h_dim, h_dim)
        self.enc3 = nn.Linear(h_dim, h_dim)

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim) # comentar esta linea para AE
        # meterle capa clasificacion jerarquica
        self.dec1 = nn.Linear(z_dim, h_dim)
        self.dec2 = nn.Linear(h_dim, h_dim)
        self.dec3 = nn.Linear(h_dim, h_dim)
        self.dec4 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
            h = F.relu(self.enc1(x))
            h = F.relu(self.enc2(h))
            h = F.relu(self.enc3(h))
            return self.fc_mu(h), self.fc_logvar(h) # return para VAE
            # return self.fc_mu(h) # return para AE

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        return self.dec4(h)

    def forward_all(self, x):
        # VAE
        mu, logvar = self.encode(x.view(-1, self.x_dim)) # flatten in case x is not flat
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z=mu
        return self.decode(z), z, mu, logvar
        # AE
        # mu = self.encode(x.view(-1, self.x_dim))
        # return self.decode(mu), mu
        

    def forward(self, x):
        return self.forward_all(x)

# Para VAE
def VAE_loss_function(recon_x, x, mu, logvar, x_dim=640):
    """Loss function for VAE which consists of reconstruction and KL divergence losses.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py

    You can also balance weights for each loss, just to see what if KLD loss is stronger, etc.

    Args:
        reconst_loss: Reconstruction loss calculation: 'mse' or 'bce'
        a_RECONST: Weight for reconstruction loss.
        a_KLD: Weight for KLD loss.
    """

    reconst_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # comentar para AE

    return reconst_loss, kld

# Para AE
def AE_loss_function(recon_x, x, x_dim=640):
    """Loss function for AE which consists of reconstruction loss only.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py

    You can also balance weights for each loss, just to see what if KLD loss is stronger, etc."""

    reconst_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')

    return reconst_loss
