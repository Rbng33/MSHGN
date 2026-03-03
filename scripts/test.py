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
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mshgn.model import MSHGN
from mshgn.data import Dataset_ETT_hour


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_mask(B, C, L, ratio, device):
    return (torch.rand(B, C, L, device=device) > ratio).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────
    root = os.path.dirname(args.data_path)
    fname = os.path.basename(args.data_path)
    seq_len = cfg['data']['seq_len']

    test_ds = Dataset_ETT_hour(
        root, 'test', [seq_len, 0, 0],
        cfg['data']['features'], fname
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    C = test_ds[0][0].shape[-1]
    log(f"Channels: {C} | Seq_len: {seq_len}")

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
        use_checkpoint=False,  # disable for inference
        use_amp=False
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    log(f"Loaded checkpoint: {args.ckpt}")
    log(f"Parameters: {model.count_parameters():,}")

    # ── Testing ──────────────────────────────────────────────
    ratios = cfg['training']['masking_ratios']
    crit = nn.MSELoss(reduction='none')

    results = {r: [] for r in ratios}

    log("\nTesting...")
    log("=" * 60)

    with torch.no_grad():
        for r in ratios:
            for i, (bx, *_) in enumerate(test_dl):
                torch.manual_seed(int(r * 1000) + i)

                bx = bx.float().to(device)
                xf = bx.transpose(1, 2).contiguous()

                B_ = xf.shape[0]
                mask = generate_mask(B_, C, seq_len, r, device)

                out = model(xf, mask=mask).float()
                mi = 1.0 - mask

                mse = (
                    (crit(out, xf) * mi).sum().item() /
                    (mi.sum().item() + 1e-8)
                )

                results[r].append(mse)

    # ── Report ───────────────────────────────────────────────
    log("\nTest Results")
    log("=" * 60)

    avg_all = []

    for r in ratios:
        avg_r = np.mean(results[r])
        avg_all.append(avg_r)
        log(f"Mask {int(r*100)}% → MSE: {avg_r:.6f}")

    final = np.mean(avg_all)
    log("-" * 60)
    log(f"Average MSE across ratios: {final:.6f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
