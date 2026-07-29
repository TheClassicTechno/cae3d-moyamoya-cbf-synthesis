#!/usr/bin/env python3
"""
3D VAE (AutoencoderKL) for Latent Diffusion
============================================
Tip #1 from difusion3dtips.txt: Use latent diffusion

This VAE compresses 3D volumes to a lower-dimensional latent space,
allowing diffusion to operate on much smaller tensors.

Architecture:
- Encoder: 3D UNet-like encoder with downsampling
- Latent space: 4-8× smaller spatially, with multiple channels
- Decoder: 3D UNet-like decoder with upsampling
- KL divergence loss for regularization

Based on MONAI AutoencoderKL patterns and difusion3dtips.txt recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.blocks import ResidualUnit
from monai.networks.layers import Norm


class VAE3D(nn.Module):
    """
    3D Variational Autoencoder for compressing volumes.
    
    Compresses volumes by 4-8× in each spatial dimension.
    For 128×128×64 input  ~32×32×16 latent (4× downsampling).
    """
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        channels=(32, 64, 128, 256),
        num_res_blocks=2,
        downsample_factor=4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        
        # Encoder: Downsample to latent space
        encoder_layers = []
        in_ch = in_channels
        
        # Calculate number of downsampling steps needed
        num_downsamples = int(np.log2(downsample_factor))
        
        # Use exactly num_downsamples channels for downsampling steps
        encoder_channels = list(channels[:num_downsamples])
        
        for i, out_ch in enumerate(encoder_channels):
            # Downsample (stride=2 halves spatial dimensions)
            encoder_layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.InstanceNorm3d(out_ch))
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Residual blocks
            for _ in range(num_res_blocks):
                encoder_layers.append(ResidualUnit(
                    spatial_dims=3,
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    norm=Norm.INSTANCE,
                    act=("LeakyReLU", {"inplace": True}),
                ))
            
            in_ch = out_ch
        
        # Final projection to latent * 2 (mean + logvar)
        encoder_layers.append(nn.Conv3d(in_ch, latent_channels * 2, kernel_size=3, padding=1))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: Upsample from latent space
        decoder_layers = []
        in_ch = latent_channels
        
        # Decoder should have same number of upsampling steps as encoder downsampling
        # Use same channels as encoder but reversed, plus one more for final output
        decoder_channels = list(reversed(encoder_channels))
        
        for i, out_ch in enumerate(decoder_channels):
            # Upsample (stride=2 to double spatial dimensions)
            decoder_layers.append(nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            
            decoder_layers.append(nn.InstanceNorm3d(out_ch))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Residual blocks
            for _ in range(num_res_blocks):
                decoder_layers.append(ResidualUnit(
                    spatial_dims=3,
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    norm=Norm.INSTANCE,
                    act=("LeakyReLU", {"inplace": True}),
                ))
            
            in_ch = out_ch
        
        # Final projection to image
        decoder_layers.append(nn.Conv3d(in_ch, in_channels, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        # Split into mean and logvar (encoder outputs latent_channels * 2)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to image space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def encode_to_latent(self, x):
        """Encode and return deterministic latent (mean only, for inference)."""
        mu, logvar = self.encode(x)
        return mu
    
    def decode_from_latent(self, z):
        """Decode from latent space."""
        return self.decode(z)


def kl_loss(mu, logvar):
    """KL divergence loss for VAE regularization."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def vae_loss(recon, target, mu, logvar, recon_weight=1.0, kl_weight=0.0001):
    """
    VAE loss: reconstruction + KL divergence.
    
    Args:
        recon: Reconstructed image
        target: Target image
        mu: Latent mean
        logvar: Latent log variance
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence (typically small)
    """
    # Reconstruction loss (L1 + L2)
    recon_loss_l1 = F.l1_loss(recon, target)
    recon_loss_l2 = F.mse_loss(recon, target)
    recon_loss = recon_loss_l1 + 0.1 * recon_loss_l2
    
    # KL divergence
    kl = kl_loss(mu, logvar)
    
    total_loss = recon_weight * recon_loss + kl_weight * kl
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl,
    }
