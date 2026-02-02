import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_VAE(nn.Module):

    def __init__(self, device, z_dim=32):
        super().__init__()
        self.device = device
        self.z_dim = z_dim

        # Encoder convolucional
        # Entrada: 1 x 128 x 311
        self.enc1 = nn.Conv2d(1, 16, kernel_size=(3,5), stride=(2,2), padding=(1,2))  # -> 16 x 64 x 156
        self.enc2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1)     # -> 32 x 32 x 78
        self.enc3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1)    # -> 64 x 16 x 39

        # Flatten para latente
        self.flatten_dim = 64 * 16 * 39
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)  # comentar para AE

        # Decoder lineal inicial para expandir desde latente
        self.dec_fc = nn.Linear(z_dim, self.flatten_dim)

        # Decoder convolucional (transposed convolutions)
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)) # -> 32 x 32 x 78
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1,1))  # -> 16 x 64 x 156
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=(3,5), stride=2, padding=(1,2), output_padding=(1,0)) # -> 1 x 128 x 311

    # -----------------
    # Encoder
    # -----------------
    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = h.view(-1, self.flatten_dim)  # flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)        # comentar para AE
        return mu, logvar                  # para VAE
        # return mu                        # para AE

    # -----------------
    # Reparametrization trick
    # -----------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # -----------------
    # Decoder
    # -----------------
    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(-1, 64, 16, 39)  # reshape a volumen
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = torch.sigmoid(self.dec3(h))  # salida entre 0 y 1
        return h

    # -----------------
    # Forward completo
    # -----------------
    def forward_all(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def forward(self, x):
        return self.forward_all(x)

# Loss
def VAE_loss_function(recon_x, x, mu, logvar):
    """Loss function for VAE which consists of reconstruction and KL divergence losses.
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld_loss
