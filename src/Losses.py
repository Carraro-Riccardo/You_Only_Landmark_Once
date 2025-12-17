import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import lpips

# -------------------------
# Pixel loss
# -------------------------
pixel_crit = nn.MSELoss()

# -------------------------
# Perceptual (VGG) loss
# -------------------------
class VGGPerceptualLoss(nn.Module):
    """
    Expects inputs in [-1,1]. Internally maps to [0,1] and applies ImageNet mean/std.
    Runs VGG feature extraction in float32 (even under autocast) for stability.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.slice1 = nn.Sequential(*list(vgg[:4])).eval()   # conv1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9])).eval()  # conv2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])).eval() # conv3_3
        for m in (self.slice1, self.slice2, self.slice3):
            for p in m.parameters():
                p.requires_grad = False

        # ImageNet norm buffers
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0,1] -> ImageNet norm
        x01 = (x.clamp(-1,1) + 1) / 2
        return (x01 - self.mean) / self.std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Force fp32 for VGG path even if outer training is mixed precision
        sr32 = self._prep(sr).float()
        hr32 = self._prep(hr).float()

        f1_sr, f1_hr = self.slice1(sr32), self.slice1(hr32)
        f2_sr, f2_hr = self.slice2(f1_sr),  self.slice2(f1_hr)
        f3_sr, f3_hr = self.slice3(f2_sr),  self.slice3(f2_hr)

        # Sum of MSEs across a few layers
        return (F.mse_loss(f1_sr, f1_hr) +
                F.mse_loss(f2_sr, f2_hr) +
                F.mse_loss(f3_sr, f3_hr))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
perceptual_crit = VGGPerceptualLoss().to(device)

def attention_loss(
    sr, hr, heat,
    *, gamma: float = 1.3,   # >1 = more focus on hot zones
       floor: float = 0.10,  # weight outside the mask (0..1)
       eps: float = 1e-6
):
    """
    Per-sample normalized masked MAE:
      loss_i = sum(w_i * |sr - hr|) / sum(w_i), then averaged over the batch.
    - sr, hr: (B, C, H, W) in [-1, 1]
    - heat: (B, 1, H, W) or (B, H, W)
    - gamma: selectivity of the heatmap
    - floor: minimum gradient even outside the mask
    """
    if heat is None:
        return sr.new_tensor(0.0)

    if heat.dim() == 3:
        heat = heat.unsqueeze(1)  # (B,1,H,W)

    heat = heat.to(device=sr.device, dtype=sr.dtype)
    B = heat.size(0)

    # min-max per campione -> [0,1]
    flat = heat.reshape(B, -1)
    hmin = flat.min(dim=1, keepdim=True)[0].reshape(B,1,1,1)
    hmax = flat.max(dim=1, keepdim=True)[0].reshape(B,1,1,1)
    span = (hmax - hmin)

    hn = (heat - hmin) / span.clamp_min(eps)     # [0,1]
    if abs(gamma - 1.0) > 1e-6:
        hn = hn.clamp(0,1).pow(gamma)

    # w' = floor + (1-floor)*hn  in [floor,1]
    w = floor + (1.0 - floor) * hn

    # se mappa ~costante, usa pesi uniformi (tutti 1)
    uniform = (span <= eps)
    if uniform.any():
        w = torch.where(uniform, torch.ones_like(w), w)

    # niente grad attraverso i pesi
    w = w.expand_as(sr).detach()

    # riduzione per-sample (reshape evita problemi di contiguità)
    w_flat   = w.reshape(B, -1)
    mae_flat = (w * (sr - hr).abs()).reshape(B, -1)
    loss_per_sample = mae_flat.sum(dim=1) / w_flat.sum(dim=1).clamp_min(eps)
    return loss_per_sample.mean()


# -------------------------
# LPIPS loss
# -------------------------
# lpips expects inputs in [-1,1]; returns (B,1,1,1) or (B,)
lpips_crit = lpips.LPIPS(net='vgg').to(device)

def lpips_loss(sr, hr):
    return lpips_crit(sr, hr).mean()