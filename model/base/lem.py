"""Low-level feature enhancing module (LEM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowLevelEnhancingModule(nn.Module):
    """Implements the Low-Level Enhancing Module (LEM).

    The module injects domain perturbations into shallow support features by
    applying a random convolution followed by an FFT-based shape preserving
    recombination step as described in
    "The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation".
    """

    def __init__(self, sigma: float = 0.1, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions")
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)

    def forward(self, Fs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if Fs.dim() != 4:
            raise ValueError("Fs must be a 4D tensor of shape [B, C, H, W]")

        C = Fs.shape[1]

        weight = torch.randn(
            C,
            C,
            self.kernel_size,
            self.kernel_size,
            device=Fs.device,
            dtype=Fs.dtype,
        ) * self.sigma

        F_prime = F.conv2d(
            Fs,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
        )

        Fs_fft = torch.fft.fft2(Fs, dim=(-2, -1))
        Fp_fft = torch.fft.fft2(F_prime, dim=(-2, -1))

        amp_prime = torch.abs(Fp_fft)
        phase_orig = torch.angle(Fs_fft)

        real = amp_prime * torch.cos(phase_orig)
        imag = amp_prime * torch.sin(phase_orig)
        new_fft = torch.complex(real, imag)

        Ft_s = torch.fft.ifft2(new_fft, dim=(-2, -1)).real
        return Ft_s

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, kernel_size={self.kernel_size}"

