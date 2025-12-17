from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MSSSIMMetric

from Model import *
from Losses import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, cfg):
        self.cfg    = cfg
        self.G = SuperResolutionUNet(
            in_channels   = cfg.get('in_channels', 3),
            base_filters  = cfg.get('base_filters', 32),
            out_channels  = cfg.get('out_channels', 3),
            refine_blocks = cfg.get('refine_blocks', 3)
        ).to(device)
        self.optG   = torch.optim.Adam(self.G.parameters(), lr=cfg['lr_g'])
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.metric_stride = 5  # compute metrics every N batches
        self.msssim_metric = MSSSIMMetric(
            data_range=2.0,
            kernel_size=(7, 7),                         # smaller window for 128×128
            betas=(0.0448, 0.2856, 0.3001, 0.2363),     # 4 scales
            normalize="relu",
        ).to(device)

    @staticmethod
    def psnr_from_mse(mse, max_val=2.0):  # max_val=2.0 for [-1,1]
        return 10.0 * torch.log10((max_val * max_val) / (mse + 1e-8))

    def compute_overall_metrics(self, sr, hr):
        """Overall PSNR/SSIM/MS-SSIM/LPIPS with inputs in [-1,1]."""
        with torch.no_grad():
            mse_overall = ((sr - hr) ** 2).mean(dim=(1,2,3))
            psnr_overall   = self.psnr_from_mse(mse_overall, max_val=2.0).mean().item()
            ssim_overall   = float(ssim(sr, hr, data_range=2.0))
            msssim_overall = float(self.msssim_metric(sr, hr).cpu())
            self.msssim_metric.reset()
            lpips_overall  = lpips_loss(sr, hr)
            if torch.is_tensor(lpips_overall):
                lpips_overall = lpips_overall.mean().item()
            return {
                'psnr_overall': psnr_overall,
                'ssim_overall': ssim_overall,
                'msssim_overall': msssim_overall,
                'lpips_overall': lpips_overall
            }

    def train_epoch(self, loader):
        agg = {
            'loss_pixel': 0.0, 'loss_perc': 0.0, 'loss_heatmaps': 0.0,
            'loss_lpips': 0.0, 'loss_combined': 0.0,
            'psnr_overall': 0.0, 'ssim_overall': 0.0,
            'msssim_overall': 0.0, 'lpips_overall': 0.0
        }
        self.G.train()
        step = 0
        metric_steps = 0
        use_cuda = torch.cuda.is_available()

        for _, _, hr, lr, heat in tqdm(loader, desc=f"Training {self.cfg['name']}"):
            lr   = lr.to(device, non_blocking=True)
            hr   = hr.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True) if self.cfg.get('use_heatmaps', False) else None

            self.optG.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_cuda, dtype=torch.float16):
                sr    = self.G(lr)
                Lpix  = pixel_crit(sr, hr)
                Lperc = perceptual_crit(sr, hr)              if self.cfg.get('use_perc',  False) else 0.0
                Lattn = attention_loss(sr, hr, heat)         if self.cfg.get('use_heatmaps',  False) else 0.0
                Llp   = lpips_loss(sr, hr)                   if self.cfg.get('use_lpips', False) else 0.0

                loss  = Lpix \
                      + self.cfg.get('w_perc',0.0)  * (Lperc if isinstance(Lperc, torch.Tensor) else 0.0) \
                      + self.cfg.get('w_heatmaps',0.0)  * (Lattn if isinstance(Lattn, torch.Tensor) else 0.0) \
                      + self.cfg.get('w_lpips',0.0) * (Llp   if isinstance(Llp,   torch.Tensor) else 0.0)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optG)
            self.scaler.update()

            agg['loss_pixel']    += float(Lpix.detach())
            agg['loss_perc']     += float(Lperc.detach()) if self.cfg.get('use_perc',  False) else 0.0
            agg['loss_heatmaps']     += float(Lattn.detach()) if self.cfg.get('use_heatmaps',  False) else 0.0
            agg['loss_lpips']    += float(Llp.detach())   if self.cfg.get('use_lpips', False) else 0.0
            agg['loss_combined'] += float(loss.detach())

            if step % self.metric_stride == 0:
                m = self.compute_overall_metrics(sr.detach(), hr)
                for k in ('psnr_overall','ssim_overall','msssim_overall','lpips_overall'):
                    agg[k] += m[k]
                metric_steps += 1

            step += 1

        out = {}
        for k in ('loss_pixel','loss_perc','loss_heatmaps','loss_lpips','loss_combined'):
            out[k] = agg[k] / step
        for k in ('psnr_overall','ssim_overall','msssim_overall','lpips_overall'):
            out[k] = agg[k] / (metric_steps if metric_steps > 0 else 1)
        return out

    def evaluate(self, loader, num_samples=500):
        self.G.eval()
        agg = {
            'loss_pixel':0.0, 'loss_perc':0.0, 'loss_lpips':0.0,
            'psnr_overall':0.0, 'ssim_overall':0.0, 'msssim_overall':0.0, 'lpips_overall':0.0
        }
        n = 0
        with torch.no_grad():
            for _, _, hr, lr, _ in tqdm(loader, desc=f"Evaluating {self.cfg['name']}"):
                if n >= num_samples:
                    break
                lr   = lr.to(device, non_blocking=True)
                hr   = hr.to(device, non_blocking=True)

                sr = self.G(lr)

                m = self.compute_overall_metrics(sr, hr)
                for k in ('psnr_overall','ssim_overall','msssim_overall','lpips_overall'):
                    agg[k] += m[k]

                Lpix  = pixel_crit(sr, hr)
                Lperc = perceptual_crit(sr, hr)              if self.cfg.get('use_perc',  False) else 0.0
                Llp   = lpips_loss(sr, hr)                   if self.cfg.get('use_lpips', False) else 0.0

                agg['loss_pixel']    += float(Lpix)
                agg['loss_perc']     += float(Lperc) if self.cfg.get('use_perc', False) else 0.0
                agg['loss_lpips']    += float(Llp)   if self.cfg.get('use_lpips', False) else 0.0

                n += 1

        return {k: (v / max(n,1)) for k, v in agg.items()}
