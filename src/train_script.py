#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tabulate import tabulate
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import h5py

from DataLoader import CelebDataSet
from Trainer import Trainer
from Losses import *
from Model import *

# ---------------------------
# Utilities
# ---------------------------
def to01(x): 
    return (x.clamp(-1,1) + 1)/2

def save_checkpoint(trainer, history, epoch, best_score, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': trainer.G.state_dict(),
        'opt': trainer.optG.state_dict(),
        'scaler': trainer.scaler.state_dict(),
        'sched': trainer.lr_sched.state_dict() if hasattr(trainer, 'lr_sched') else None,
        'early_best': best_score,
        'history': history,
    }, path)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta
        self.best=None; self.bad_epochs=0; self.should_stop=False
    def step(self, score):
        if self.best is None or score < self.best - self.min_delta:
            self.best=score; self.bad_epochs=0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience: self.should_stop=True

def val_objective(m, lp_weight, ssim_weight, ssim_target, psnr_weight, psnr_target):
    ssim = float(m.get('ssim_overall', 0.0))
    lp   = float(m.get('lpips_overall', 1.0))
    psnr = float(m.get('psnr_overall', 0.0))
    return (lp_weight*lp) + (ssim_weight*max(0.0, ssim_target-ssim)) + (psnr_weight*max(0.0, psnr_target-psnr))

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Super-resolution training with heatmap guidance — CLI configurable"
    )

    # Paths & data
    p.add_argument("--data_path", type=str, required=True,
                   help="CelebA root directory (expects split info in your dataset class).")
    p.add_argument("--heat_h5", type=str, required=True,
                   help="Path to heatmaps HDF5 (dataset key 'heatmaps').")
    p.add_argument("--out_dir_models", type=str, required=True,
                   help="Output directory for checkpoints")
    p.add_argument("--out_dir_figures", type=str, required=True,
                   help="Output directory for figures")
    p.add_argument("--train_limit", type=int, default=None,
                   help="Max training samples (None=use all available).")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Validation portion relative to train_limit (0..1).")
    p.add_argument("--num_workers", type=int, default=2)

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto","cpu","cuda"])
    p.add_argument("--pin_memory", action="store_true", help="Pin dataloader memory (CUDA).")
    p.add_argument("--no_pin_memory", action="store_true", help="Force pin_memory=False.")
    p.add_argument("--deterministic", action="store_true", help="Enable cudnn deterministic.")

    # Model/Trainer config
    p.add_argument("--name", type=str, default="base_heat_128_batch_no_schedule")
    p.add_argument("--base_filters", type=int, default=48)
    p.add_argument("--refine_blocks", type=int, default=5)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--use_bicubic", action="store_true", help="Use bicubic interpolation in data loader.")

    # Loss toggles
    p.add_argument("--use_perc", action="store_true")
    p.add_argument("--use_heatmaps", action="store_true")
    p.add_argument("--use_lpips", action="store_true")
    p.add_argument("--w_perc", type=float, default=0.01)
    p.add_argument("--w_heatmaps", type=float, default=2.0)
    p.add_argument("--w_lpips", type=float, default=0.15)

    # LR scheduler & early stop
    p.add_argument("--plateau_patience", type=int, default=6,
                   help="ReduceLROnPlateau patience.")
    p.add_argument("--plateau_factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor.")
    p.add_argument("--early_patience", type=int, default=15)
    p.add_argument("--early_min_delta", type=float, default=1e-4)

    # Loss 
    p.add_argument("--target_psnr", default=26.0, type=float,
                   help="Target PSNR for validation stopping criterion.")
    p.add_argument("--target_ssim", default=0.85, type=float,
                   help="Target SSIM for validation stopping criterion.")

    p.add_argument("--ssim_weight", default=0.8, type=float,
                   help="SSIM weight for validation stopping criterion.")
    p.add_argument("--psnr_weight", default=0.5, type=float,
                   help="PSNR weight for validation stopping criterion.")
    p.add_argument("--lpips_weight", default=1.5, type=float,
                     help="LPIPS weight for validation stopping criterion.")
    # Checkpointing
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Resume full trainer checkpoint (.pt).")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--viz_every", type=int, default=5,
                   help="Save reconstruction grid every N epochs.")

    return p.parse_args()

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # ----- device & seeds
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pin = False
    if args.no_pin_memory:
        pin = False
    else:
        pin = args.pin_memory and (device.type == "cuda")

    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ----- heatmap count
    with h5py.File(args.heat_h5, 'r') as f:
        HM_COUNT = int(f['heatmaps'].shape[0])
    print("Heatmaps available:", HM_COUNT)

    # ----- limits
    train_limit = args.train_limit if args.train_limit is not None else HM_COUNT
    val_limit   = max(1, int(args.val_split * train_limit))

    # ----- datasets
    train_ds = CelebDataSet(args.data_path, 'train', heatmap_h5=args.heat_h5, use_bicubic=args.use_bicubic)
    val_ds   = CelebDataSet(args.data_path, 'val',   heatmap_h5=args.heat_h5, use_bicubic=args.use_bicubic)

    train_n = min(train_limit, len(train_ds))
    val_n   = min(val_limit,   len(val_ds))

    print(f"Using TRAIN_LIMIT={train_n}, VAL_LIMIT={val_n}")
    train_subset = Subset(train_ds, range(train_n))
    val_subset   = Subset(val_ds,   range(val_n))

    # ----- dataloaders
    loader     = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    print("Train samples:", len(train_subset))
    print("Val samples:  ", len(val_subset))

    # ----- one fixed batch for viz
    data_iter = iter(loader)
    _, _, hr_f, lr_f, heat_f = next(data_iter)
    lr_vis, hr_vis, heat_vis = [t.to(device, non_blocking=True) for t in (lr_f[0:1], hr_f[0:1], heat_f[0:1])]

    # ----- config (single, via CLI)
    schedule_flag = args.schedule and not args.no_schedule

    cfg = dict(
        name=args.name,
        use_heatmaps=args.use_heatmaps, use_perc=args.use_perc, use_lpips=args.use_lpips,
        w_heatmaps=args.w_heatmaps, w_perc=args.w_perc, w_lpips=args.w_lpips,
        lr_g=args.lr_g,
        base_filters=args.base_filters,
        refine_blocks=args.refine_blocks,
        schedule=schedule_flag
    )
    # store base weights for scheduler
    for k in ('w_perc','w_heatmaps','w_lpips'):
        cfg.setdefault(k, 0.0)
        cfg.setdefault(f'_base_{k}', cfg[k])

    # ----- build trainer
    t = Trainer(cfg)
    t.lr_sched = ReduceLROnPlateau(t.optG, mode='min',
                                   patience=args.plateau_patience,
                                   factor=args.plateau_factor,
                                   verbose=True)

    # ----- early stop
    early_stop = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)
    history    = {}
    start_epoch = 1

    # ----- resume full checkpoint (optional)
    best_score = float('inf')
    best_dir = os.path.join(args.out_dir_models, cfg['name'])
    os.makedirs(best_dir, exist_ok=True)
    best_path = os.path.join(best_dir, f"{cfg['name']}_best.pth")

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        t.G.load_state_dict(ckpt['model'])
        t.optG.load_state_dict(ckpt['opt'])
        t.scaler.load_state_dict(ckpt['scaler'])
        if ckpt.get('sched'):
            t.lr_sched.load_state_dict(ckpt['sched'])
        print("Loaded configuration from", args.ckpt_path)

        early_stop.best = ckpt.get('early_best', float('inf'))
        t._best_score = early_stop.best
        best_score = early_stop.best
        start_epoch = ckpt.get('epoch', 0) + 1
        history = ckpt.get('history', {})

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    for epoch in range(start_epoch, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # ---- TRAIN ----
        metrics = t.train_epoch(loader)
        # history (train)
        for k, v in metrics.items():
            history.setdefault(k, []).append(v)

        # print train
        headers = ["Model","pix","perc","attn","lpips_loss","comb","PSNR","SSIM","MS-SSIM","LPIPS"]
        row = [
            cfg['name'],
            f"{metrics['loss_pixel']:.4e}",
            f"{metrics['loss_perc']:.4e}"    if cfg.get('use_perc', False)  else "-",
            f"{metrics['loss_heatmaps']:.4e}"    if cfg.get('use_heatmaps', False)  else "-",
            f"{metrics['loss_lpips']:.4e}"   if cfg.get('use_lpips', False) else "-",
            f"{metrics['loss_combined']:.4e}",
            f"{metrics['psnr_overall']:.2f}",
            f"{metrics['ssim_overall']:.4f}",
            f"{metrics['msssim_overall']:.4f}",
            f"{metrics['lpips_overall']:.4f}",
        ]
        print(tabulate([row], headers=headers, tablefmt="github"))

        # ---- VALIDATION ----
        vm = t.evaluate(val_loader)
        for k, v in vm.items():
            history.setdefault('val_' + k, []).append(v)

        val_headers = ["Model","pix","perc","lpips_loss","PSNR","SSIM","MS-SSIM","LPIPS","val_obj"]
        vrow = [
            cfg['name'],
            f"{vm['loss_pixel']:.4e}",
            f"{vm['loss_perc']:.4e}"    if cfg.get('use_perc', False)  else "-",
            f"{vm['loss_lpips']:.4e}"   if cfg.get('use_lpips', False) else "-",
            f"{vm['psnr_overall']:.2f}",
            f"{vm['ssim_overall']:.4f}",
            f"{vm['msssim_overall']:.4f}",
            f"{vm['lpips_overall']:.4f}",
            f"{val_objective(vm, args.lpips_weight, args.ssim_weight, args.target_ssim, args.psnr_weight, args.target_psnr):.4f}",
        ]
        print("\n")
        print(tabulate([vrow], headers=val_headers, tablefmt="github"))
    
        score = val_objective(vm, args.lpips_weight, args.ssim_weight, args.target_ssim, args.psnr_weight, args.target_psnr)

        fig_dir = os.path.join(args.out_dir_figures, cfg['name'], "figures")
        os.makedirs(fig_dir, exist_ok=True)
        results_path = os.path.join(fig_dir, "results.txt")

        with open(results_path, "a") as f:
            f.write(
                f"\n{epoch},"
                f"{metrics['loss_pixel']:.6f},"
                f"{metrics['loss_perc']:.6f}"    if cfg.get('use_perc', False) else "-" + ","
                f"{metrics['loss_heatmaps']:.6f}"    if cfg.get('use_heatmaps', False) else "-" + ","
                f"{metrics['loss_lpips']:.6f}"   if cfg.get('use_lpips', False) else "-" + ","
                f"{metrics['loss_combined']:.6f},"
                f"{metrics['psnr_overall']:.4f},"
                f"{metrics['ssim_overall']:.6f},"
                f"{metrics['msssim_overall']:.6f},"
                f"{metrics['lpips_overall']:.6f}\n"
            )

        with open(results_path, "a") as f:
            f.write(
                f"\nval_{epoch},"
                f"{vm['loss_pixel']:.6f},"
                f"{vm['loss_perc']:.6f}"    if cfg.get('use_perc', False) else "-" + ","
                f"{vm['loss_lpips']:.6f}"   if cfg.get('use_lpips', False) else "-" + ","
                f"{vm['psnr_overall']:.4f},"
                f"{vm['ssim_overall']:.6f},"
                f"{vm['msssim_overall']:.6f},"
                f"{vm['lpips_overall']:.6f},"
                f"{score:.6f}\n"
            )

        # schedulers + early stop
        t.lr_sched.step(score)
        early_stop.step(score)

        # quick viz
        if args.viz_every > 0 and (epoch % args.viz_every == 0):
            val_iter = iter(val_loader)
            _, _, hr_val_b, lr_val_b, _ = next(val_iter)
            lr_val = lr_val_b[:5].to(device)
            hr_val = hr_val_b[:5].cpu()

            lr_up_val = to01(F.interpolate(lr_val, size=(128,128), mode='bilinear', align_corners=False)).cpu()
            with torch.no_grad():
                sr = t.G.eval()(lr_val)
                recon = to01(sr).cpu()

            fig, axes = plt.subplots(5, 3, figsize=(12, 20))
            for i in range(5):
                axes[i,0].imshow(lr_up_val[i].permute(1,2,0)); axes[i,0].set_title("LR ↑"); axes[i,0].axis('off')
                axes[i,1].imshow(to01(hr_val[i]).permute(1,2,0)); axes[i,1].set_title("HR GT"); axes[i,1].axis('off')
                axes[i,2].imshow(recon[i].permute(1,2,0)); axes[i,2].set_title(cfg['name']); axes[i,2].axis('off')
            plt.suptitle(f"Epoch {epoch} — Validation Reconstructions", fontsize=16)
            plt.tight_layout()
            fig_dir = os.path.join(args.out_dir_figures, cfg['name'], "figures"); os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"val_epoch_{epoch}.png"))
            plt.close(fig)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak VRAM: {peak:.2f} GB")

        # save best + periodic
        if score <= getattr(t, "_best_score", float("inf")):
            torch.save(t.G.state_dict(), best_path)
            t._best_score = score

        if args.save_every > 0 and (epoch % args.save_every == 0):
            path_to_save = os.path.join(args.out_dir_models, cfg['name'], f"{cfg['name']}_epoch{epoch:03d}.pt")
            save_checkpoint(t, history, epoch, early_stop.best, path_to_save)

        if early_stop.should_stop:
            print(f"\nEarly stopping at epoch {epoch} (best val objective: {early_stop.best:.4f}).")
            # load best model state_dict
            best_sd = torch.load(best_path, map_location=device)
            t.G.load_state_dict(best_sd)

            # recompute validation metrics for best model
            best_metrics = t.evaluate(val_loader)

            print("\n=== Best Model Validation Metrics ===")
            print(tabulate([[
                cfg['name'],
                f"{best_metrics['loss_pixel']:.4e}",
                f"{best_metrics['loss_perc']:.4e}"    if cfg.get('use_perc', False)  else "-",
                f"{best_metrics['loss_lpips']:.4e}"   if cfg.get('use_lpips', False) else "-",
                f"{best_metrics['psnr_overall']:.2f}",
                f"{best_metrics['ssim_overall']:.4f}",
                f"{best_metrics['msssim_overall']:.4f}",
                f"{best_metrics['lpips_overall']:.4f}",
                f"{val_objective(best_metrics, args.lpips_weight, args.ssim_weight, args.target_ssim, args.psnr_weight, args.target_psnr):.4f}",
            ]], headers=["Model","pix","perc","lpips_loss","PSNR","SSIM","MS-SSIM","LPIPS","val_obj"], tablefmt="github"))

            # visualize best model
            val_iter = iter(val_loader)
            _, _, hr_val_b, lr_val_b, _ = next(val_iter)
            lr_val = lr_val_b[:5].to(device)
            hr_val = hr_val_b[:5].cpu()

            lr_up_val = to01(F.interpolate(lr_val, size=(128,128), mode='bilinear', align_corners=False)).cpu()
            with torch.no_grad():
                sr = t.G.eval()(lr_val)
                recon = to01(sr).cpu()

            fig, axes = plt.subplots(5, 3, figsize=(12, 20))
            for i in range(5):
                axes[i,0].imshow(lr_up_val[i].permute(1,2,0)); axes[i,0].set_title("LR ↑"); axes[i,0].axis('off')
                axes[i,1].imshow(to01(hr_val[i]).permute(1,2,0)); axes[i,1].set_title("HR GT"); axes[i,1].axis('off')
                axes[i,2].imshow(recon[i].permute(1,2,0)); axes[i,2].set_title(f"{cfg['name']} (best)"); axes[i,2].axis('off')
            plt.suptitle("Best Model — Validation Reconstructions", fontsize=16)
            plt.tight_layout()
            fig_dir = os.path.join(args.out_dir_figures, cfg['name'], "figures"); os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, "BestModel.png"))
            plt.close(fig)
            break

if __name__ == "__main__":
    main()
