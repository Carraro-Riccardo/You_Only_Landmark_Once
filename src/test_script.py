import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tabulate import tabulate
from ptflops import get_model_complexity_info

from DataLoader import CelebDataSet

def val_objective(m, args):
    """
    Calculates the validation objective based on provided weights and targets.
    formula: (lp_weight*lp) + (ssim_weight*max(0.0, ssim_target-ssim)) + (psnr_weight*max(0.0, psnr_target-psnr))
    """
    ssim = float(m.get('ssim_overall', 0.0))
    lp   = float(m.get('lpips_overall', 1.0))
    psnr = float(m.get('psnr_overall', 0.0))
    
    score = (args.lpips_weight * lp) + \
            (args.ssim_weight * max(0.0, args.target_ssim - ssim)) + \
            (args.psnr_weight * max(0.0, args.target_psnr - psnr))
    
    return score

def to01(x): 
    return (x.clamp(-1,1) + 1)/2

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Super Resolution Model on CelebA")

    # --- Paths ---
    parser.add_argument("--data_path", type=str, default="PATH_TO_CELEBA",
                        help="Root path to CelebA dataset.")
    parser.add_argument("--heat_h5", type=str, default="PATH_TO_HEATMAPS_H5",
                        help="Path to heatmaps H5 file.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the model checkpoint (.pth) to load.")
    parser.add_argument("--output_dir", type=str, default="PATH_TO_OUTPUT_DIR",
                        help="Directory to save output visualization.")

    # --- Model Architecture & Config ---
    parser.add_argument("--model_name", type=str, default="MODEL_NAME",
                        help="Name of the model configuration.")
    parser.add_argument("--base_filters", type=int, default=48,
                        help="Number of base filters in the model.")
    parser.add_argument("--refine_blocks", type=int, default=3,
                        help="Number of refinement blocks.")
    
    # --- Model Flags (Booleans) ---
    parser.add_argument("--use_heatmaps", action="store_true", help="Enable heatmaps mechanisms.")
    parser.add_argument("--use_perc", action="store_true", help="Enable perceptual loss usage in config.")
    parser.add_argument("--use_lpips", action="store_true", help="Enable LPIPS usage in config.")
    parser.add_argument("--no_schedule", action="store_false", dest="schedule", 
                        help="Disable scheduler (default is enabled).")
    
    # --- Loss Weights (for config reconstruction) ---
    parser.add_argument("--w_heatmaps", type=float, default=2.0, help="Weight for attention loss.")
    parser.add_argument("--w_perc", type=float, default=0.005, help="Weight for perceptual loss.")
    parser.add_argument("--w_lpips", type=float, default=0.05, help="Weight for LPIPS loss.")
    parser.add_argument("--lr_g", type=float, default=0.00007, help="Generator learning rate.")

    # --- Validation Objective Parameters ---
    parser.add_argument("--target_psnr", default=26.0, type=float,
                        help="Target PSNR for validation stopping criterion.")
    parser.add_argument("--target_ssim", default=0.85, type=float,
                        help="Target SSIM for validation stopping criterion.")
    parser.add_argument("--ssim_weight", default=0.8, type=float,
                        help="SSIM weight for validation stopping criterion.")
    parser.add_argument("--psnr_weight", default=0.5, type=float,
                        help="PSNR weight for validation stopping criterion.")
    parser.add_argument("--lpips_weight", default=1.5, type=float,
                        help="LPIPS weight for validation stopping criterion.")
    parser.add_argument("--use_bicubic", action="store_true",
                        help="Use bicubic interpolation for validation objective calculation.")

    # --- DataLoader ---
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers.")

    return parser.parse_args()

def main():
    args = get_args()

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = torch.cuda.is_available()
    print(f"Using device: {device}")

    # --- Seeds ---
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Data Loader ---
    print(f"Loading data from: {args.data_path}")
    test_ds = CelebDataSet(args.data_path, 'test', heatmap_h5=args.heat_h5, use_bicubic=args.use_bicubic)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=pin)
    print(f"Test samples: {len(test_ds)}")

    current_config = {
        'name': args.model_name,
        'use_heatmaps': args.use_heatmaps,
        'use_perc': args.use_perc,
        'use_lpips': args.use_lpips,
        'w_heatmaps': args.w_heatmaps,
        'w_perc': args.w_perc,
        'w_lpips': args.w_lpips,
        'lr_g': args.lr_g,
        'base_filters': args.base_filters,
        'refine_blocks': args.refine_blocks,
        'schedule': args.schedule
    }

    # Ensure defaults like in the original loop
    for k in ('w_perc','w_heatmaps','w_lpips'):
        current_config.setdefault(f'_base_{k}', current_config[k])

    # --- Initialize Trainer & Model ---
    trainer = Trainer(current_config)
    
    # --- Load Checkpoint ---
    print(f"Loading checkpoint: {args.ckpt_path}")
    try:
        best_ckpt = torch.load(args.ckpt_path, map_location=device)
        trainer.G.load_state_dict(best_ckpt)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        if isinstance(best_ckpt, dict) and 'state_dict' in best_ckpt:
            trainer.G.load_state_dict(best_ckpt['state_dict'])
        else:
            print("Could not load state dict. Check checkpoint structure.")
            return

    # --- Model Statistics ---
    num_params = sum(p.numel() for p in trainer.G.parameters() if p.requires_grad)
    print(f"Model [{args.model_name}] - # trainable parameters: {num_params}")

    try:
        macs, params = get_model_complexity_info(
            trainer.G,
            (3, 16, 16), # Input size
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
        print("MACs:", macs)
        print("Params:", params)
    except Exception as e:
        print(f"Complexity check failed (optional): {e}")

    # --- Evaluation ---
    print("\nStarting Evaluation...")
    best_metrics = trainer.evaluate(test_loader)
    
    # Calculate Objective
    obj_score = val_objective(best_metrics, args)

    print("\n=== Best Model Test Metrics ===")
    headers = ["Model", "pix", "perc", "lpips_loss", "PSNR", "SSIM", "MS-SSIM", "LPIPS", "val_obj"]
    
    row = [
        args.model_name,
        f"{best_metrics.get('loss_pixel', 0):.4e}",
        f"{best_metrics.get('loss_perc', 0):.4e}" if args.use_perc else "-",
        f"{best_metrics.get('loss_lpips', 0):.4e}" if args.use_lpips else "-",
        f"{best_metrics.get('psnr_overall', 0):.2f}",
        f"{best_metrics.get('ssim_overall', 0):.4f}",
        f"{best_metrics.get('msssim_overall', 0):.4f}",
        f"{best_metrics.get('lpips_overall', 0):.4f}",
        f"{obj_score:.4f}",
    ]
    
    print(tabulate([row], headers=headers, tablefmt="github"))

    # --- Visualization ---
    print("\nGenerating visualizations...")
    # Get a batch
    test_iter = iter(test_loader)
    _, _, hr_test_b, lr_test_b, _ = next(test_iter)
    
    # Select first 5 images
    lr_test = lr_test_b[:5].to(device)
    hr_test = hr_test_b[:5].cpu()

    # Simple Bilinear Upsample for comparison (or bicubic if specified)
    mode = 'bicubic' if args.use_bicubic else 'bilinear'
    lr_up_test = to01(F.interpolate(lr_test, size=(128,128), mode=mode, align_corners=False)).cpu()
    
    # Model Prediction
    with torch.no_grad():
        trainer.G.eval()
        sr = trainer.G(lr_test)
        recon_test = to01(sr).cpu()

    # Plotting
    # Columns: LR, HR, Prediction
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    
    for i in range(5):
        # LR
        axes[i, 0].imshow(lr_up_test[i].permute(1,2,0))
        axes[i, 0].set_title("LR ↑")
        axes[i, 0].axis('off')
        
        # HR
        axes[i, 1].imshow(to01(hr_test[i]).permute(1,2,0))
        axes[i, 1].set_title("HR GT")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(recon_test[i].permute(1,2,0))
        axes[i, 2].set_title(f"{args.model_name}")
        axes[i, 2].axis('off')

    plt.suptitle(f"Model: {args.model_name} — Test Reconstructions", fontsize=16)
    plt.tight_layout()
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"Test_{args.model_name}.png")
    plt.savefig(save_path)
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    main()