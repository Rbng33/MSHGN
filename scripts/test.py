"""
MSHGN Test Script
Usage: python scripts/test.py --config configs/etth1_v1.yaml
                               --data_path /path/to/ETTh1.csv
                               --ckpt mshgn_best.pth
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import defaultdict
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mshgn.model import MSHGN
from mshgn.data import Dataset_ETT_hour, Dataset_ETT_minute

DATASET_MAP = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_mask(B, C, L, ratio, device):
    return (torch.rand(B, C, L, device=device) > ratio).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--ckpt',      required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────
    root    = os.path.dirname(args.data_path)
    fname   = os.path.basename(args.data_path)
    seq_len = cfg['data']['seq_len']

    dataset_name = cfg['data'].get('dataset', 'ETTh1')
    DatasetClass = DATASET_MAP.get(dataset_name, Dataset_ETT_hour)
    freq = cfg['data'].get('freq', 'h')
    log(f"Dataset class: {DatasetClass.__name__} (freq={freq})")

    test_ds = DatasetClass(
        root, 'test', [seq_len, 0, 0],
        cfg['data']['features'], fname, freq=freq
    )
    # FIX: drop_last=False to evaluate on all test samples
    test_dl = DataLoader(
        test_ds, batch_size=64,
        shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True
    )

    C = test_ds[0][0].shape[-1]
    log(f"Test samples: {len(test_ds):,} | Channels: {C} | Seq_len: {seq_len}")

    # ── Model ────────────────────────────────────────────────
    m_cfg = cfg['model']
    model = MSHGN(
        num_channels=C,
        seq_len=seq_len,
        d_model=m_cfg['d_model'],
        num_layers=m_cfg['num_layers'],
        num_scales=m_cfg['num_scales'],
        conv_kernel=m_cfg['conv_kernel'],
        n_heads=m_cfg['n_heads'],
        dropout=m_cfg['dropout'],
        use_checkpoint=False,   # disabled for inference
        use_amp=False
    ).to(device)

    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    log(f"Loaded: {args.ckpt}")
    log(f"Parameters: {model.count_parameters():,}")

    # ── Testing ──────────────────────────────────────────────
    ratios  = cfg['training']['masking_ratios']
    crit    = nn.MSELoss(reduction='none')
    mae_fn  = nn.L1Loss(reduction='none')

    # Per-ratio results
    ratio_results = {}

    # Per-channel accumulators (over all ratios combined)
    ch_mse_all  = defaultdict(float)
    ch_mae_all  = defaultdict(float)
    ch_pts_all  = defaultdict(int)

    log("\nTesting per masking rate...")
    log("=" * 60)

    with torch.no_grad():
        for r in ratios:
            mse_sum, mae_sum, n_pts = 0.0, 0.0, 0
            ch_mse = defaultdict(float)
            ch_mae = defaultdict(float)
            ch_pts = defaultdict(int)

            # Fixed seed per ratio for reproducibility
            torch.manual_seed(int(r * 10000) + 99999)

            for bx, *_ in test_dl:
                bx  = bx.float().to(device, non_blocking=True)
                xf  = bx.transpose(1, 2).contiguous()       # (B, C, L)
                B_  = xf.shape[0]
                mask = generate_mask(B_, C, seq_len, r, device)

                out  = model(xf, mask=mask).float()
                mi   = 1.0 - mask                           # missing positions

                mse  = crit(out, xf)  * mi
                mae  = mae_fn(out, xf) * mi

                mse_sum += mse.sum().item()
                mae_sum += mae.sum().item()
                n_pts   += mi.sum().item()

                # Per-channel accumulation
                for c in range(C):
                    cm = mi[:, c, :]
                    if cm.sum() > 0:
                        ch_mse[c] += mse[:, c, :].sum().item()
                        ch_mae[c] += mae[:, c, :].sum().item()
                        ch_pts[c] += int(cm.sum().item())
                        ch_mse_all[c] += mse[:, c, :].sum().item()
                        ch_mae_all[c] += mae[:, c, :].sum().item()
                        ch_pts_all[c] += int(cm.sum().item())

            avg_mse = mse_sum / (n_pts + 1e-8)
            avg_mae = mae_sum / (n_pts + 1e-8)
            ratio_results[r] = {
                'mse': avg_mse,
                'mae': avg_mae,
                'per_ch': {
                    c: {
                        'mse': ch_mse[c] / (ch_pts[c] + 1e-8),
                        'mae': ch_mae[c] / (ch_pts[c] + 1e-8)
                    }
                    for c in range(C) if ch_pts[c] > 0
                }
            }
            log(f"  Mask {int(r*100):3d}%  →  MSE: {avg_mse:.6f}  |  MAE: {avg_mae:.6f}")

    # ── Summary ──────────────────────────────────────────────
    avg_mse = np.mean([ratio_results[r]['mse'] for r in ratios])
    avg_mae = np.mean([ratio_results[r]['mae'] for r in ratios])

    log("\n" + "=" * 60)
    log(f"  {'Ratio':<10} {'MSE':<14} {'MAE':<14}")
    log(f"  {'-'*38}")
    for r in ratios:
        log(f"  {int(r*100):3d}%       "
            f"{ratio_results[r]['mse']:<14.6f} "
            f"{ratio_results[r]['mae']:<14.6f}")
    log(f"  {'-'*38}")
    log(f"  {'AVERAGE':<10} {avg_mse:<14.6f} {avg_mae:<14.6f}")
    log("=" * 60)
    log(f"  Ref HGTS-Former ETTh1: MSE=0.085  MAE=0.192")
    log("=" * 60)

    # ── Per-Channel Summary ───────────────────────────────────
    log("\nPer-Channel Results (averaged over all masking rates):")
    log(f"  {'Channel':<10} {'MSE':<14} {'MAE':<14}")
    log(f"  {'-'*38}")
    for c in range(C):
        if ch_pts_all[c] > 0:
            c_mse = ch_mse_all[c] / ch_pts_all[c]
            c_mae = ch_mae_all[c] / ch_pts_all[c]
            log(f"  Ch_{c:<7} {c_mse:<14.6f} {c_mae:<14.6f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
