import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

import common as com

class CNN_VAE(nn.Module):

    def __init__(self, device, n_mels, n_frames, z_dim=32, vae = True):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.flatten_dim = 64*((n_mels+7)//8)*((n_frames+7)//8) # hace ceil para evitar problemas ocn impares
        # self.flatten_dim = 64*n_mels//8*n_frames//8
        self.vae = vae
        # Encoder convolucional
        # Entrada: 1 x 128 x 311
        self.enc1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1))  # -> 16 x nmels/2 (64) x nframes/2 (156)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1)     # -> 32 x nmels/4 x nframes/4
        self.enc3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1)    # -> 64 x nmels/8 x nframes/8

        # Flatten para latente
        # self.flatten_dim = 64 * 16 * 39
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        if self.vae:
            self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)  # comentar para AE

        # Decoder lineal inicial para expandir desde latente
        self.dec_fc = nn.Linear(z_dim, self.flatten_dim)
        # dropout2d()
        # Decoder convolucional (transposed convolutions)
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)) # -> 32 x 32 x 78
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1,1))  # -> 16 x 64 x 156
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,1)) # -> 1 x 128 x 311

        # self.dropout = nn.Dropout2d(p=0.05)
    # -----------------
    # Encoder
    # -----------------
    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        # h = self.dropout(h)
        h = F.relu(self.enc3(h))
        # h = h.view(-1, self.flatten_dim)  # flatten
        self.shape_before_flatten = h.shape[1:]
        h = torch.flatten(h,1)
        mu = self.fc_mu(h)
        if self.vae:
        # if True:
            logvar = self.fc_logvar(h)        # comentar para AE
            return mu, logvar                  # para VAE
        else:
            return mu                       # para AE

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
        # h = h.view(-1, 64, 16, 39)  # reshape a volumen
        h = torch.unflatten(h, 1, self.shape_before_flatten)
        h = F.relu(self.dec1(h))
        # h = self.dropout(h)
        h = F.relu(self.dec2(h))
        h = self.dec3(h)  # sin sigmoid, los datos estandarizados pueden tener valores negativos
        return h

    # -----------------
    # Forward completo
    # -----------------
    def forward_all(self, x):
        # VAE
        if self.vae:
        # if True:
            mu, logvar = self.encode(x)
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                # z = self.reparameterize(mu, logvar)
                # z = self.reparameterize(mu*0, logvar*0)
                z = mu
            return self.decode(z), z, mu, logvar
        # AE
        else:
            mu = self.encode(x)
            return self.decode(mu), mu

    def forward(self, x):
        return self.forward_all(x)

# Loss
def VAE_loss_function(recon_x, x, mu, logvar):
    """Loss function for VAE which consists of reconstruction and KL divergence losses.
    """
    # Reconstruction loss puedo usar mse, smooth_l1_loss o l1_loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # recon_loss = com.cross_correlation_loss(x,recon_x,max_df=10,max_dt=4,freq_scale=0.9)
    # recon_loss = 1-ssim(recon_x, x, data_range=6.0)
    # KL Divergence loss F.kl_div
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld_loss

def AE_loss_function(recon_x, x):
    """Loss function for AE which is just the reconstruction loss.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # recon_loss = com.cross_correlation_loss(x,recon_x,max_df=10,max_dt=4,freq_scale=0.2)
    # recon_loss = 1-ssim(recon_x, x, data_range=6.0)
    recon_loss = 0.5*recon_loss+0.5*(1-ssim(recon_x,x,data_range=6.0))

    return recon_loss
