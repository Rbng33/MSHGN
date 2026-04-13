"""
MSHGN Training Script
Usage: python scripts/train.py --config configs/etth1_v1.yaml
                                --data_path /path/to/ETTh1.csv
"""
import argparse
import gc
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mshgn.model import MSHGN
from mshgn.data import Dataset_ETT_hour, Dataset_ETT_minute

DATASET_MAP = {
    'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
}


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress_bar(current, total, loss, t0, bar_width=30):
    pct = current / total
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    elapsed = time.time() - t0
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    print(
        f"\r  [{bar}] {current}/{total} "
        f"loss={loss:.4f}  "
        f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  ",
        end="", flush=True
    )


def generate_mask(B, C, L, ratio, device):
    return (torch.rand(B, C, L, device=device) > ratio).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--save_dir',  default='.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── GPU setup ─────────────────────────────────────────────
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"GPU: {torch.cuda.get_device_name(0)} | "
            f"{total/1e9:.1f}GB total | {free/1e9:.1f}GB free")

    # ── Datasets ──────────────────────────────────────────────
    root  = os.path.dirname(args.data_path)
    fname = os.path.basename(args.data_path)
    seq_len = cfg['data']['seq_len']

    dataset_name = cfg['data'].get('dataset', 'ETTh1')
    DatasetClass = DATASET_MAP.get(dataset_name, Dataset_ETT_hour)
    freq = cfg['data'].get('freq', 'h')
    log(f"Dataset: {dataset_name} | Class: {DatasetClass.__name__} | freq={freq}")

    trn_ds = DatasetClass(root, 'train', [seq_len, 0, 0],
                          cfg['data']['features'], fname, freq=freq)
    val_ds = DatasetClass(root, 'val',   [seq_len, 0, 0],
                          cfg['data']['features'], fname, freq=freq)

    C = trn_ds[0][0].shape[-1]
    log(f"Channels: {C} | Seq_len: {seq_len} | "
        f"Train: {len(trn_ds):,} samples | Val: {len(val_ds):,} samples")

    # ── Model ─────────────────────────────────────────────────
    m_cfg   = cfg['model']
    use_amp = (device != 'cpu')
    model   = MSHGN(
        num_channels=C, seq_len=seq_len,
        d_model=m_cfg['d_model'], num_layers=m_cfg['num_layers'],
        num_scales=m_cfg['num_scales'], conv_kernel=m_cfg['conv_kernel'],
        n_heads=m_cfg['n_heads'], dropout=m_cfg['dropout'],
        use_checkpoint=m_cfg['use_checkpoint'], use_amp=use_amp
    ).to(device)
    log(f"Parameters: {model.count_parameters():,}")

    # ── Training setup ────────────────────────────────────────
    t_cfg    = cfg['training']
    ratios   = t_cfg['masking_ratios']
    n_concat = t_cfg.get('n_concat', 1)
    n_passes = len(ratios) // n_concat
    groups   = [ratios[i:i+n_concat] for i in range(0, len(ratios), n_concat)]

    bs = t_cfg['batch_size']
    trn_dl = DataLoader(trn_ds, bs, shuffle=True,  drop_last=True,
                        num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, bs, shuffle=False, drop_last=True,
                        num_workers=8, pin_memory=True)

    opt    = torch.optim.AdamW(model.parameters(),
                               lr=t_cfg['lr'], weight_decay=t_cfg['weight_decay'])
    sched  = CosineAnnealingLR(opt, T_max=t_cfg['epochs'],
                               eta_min=t_cfg['eta_min'])
    crit   = nn.MSELoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Pre-generate fixed validation masks
    log("Pre-generating validation masks...")
    vmasks = {}
    for r in ratios:
        vmasks[r] = {}
        for i, (bx, *_) in enumerate(val_dl):
            torch.manual_seed(int(r * 1000) + i)
            vmasks[r][i] = generate_mask(bx.shape[0], C, seq_len, r, 'cpu')

    best_val, patience_cnt = float('inf'), 0
    ckpt      = os.path.join(args.save_dir, 'mshgn_best.pth')
    n_batches = len(trn_dl)

    log(f"\nTraining {t_cfg['epochs']} epochs | "
        f"LR={t_cfg['lr']} | BS={bs} | {n_batches} batches/epoch")
    log("=" * 70)

    for ep in range(t_cfg['epochs']):
        ep_t0 = time.time()

        # ── Train ──
        model.train()
        losses = []

        for bi, (bx, *_) in enumerate(trn_dl):
            bx = bx.float().to(device, non_blocking=True)
            xf = bx.transpose(1, 2).contiguous()
            B_ = xf.shape[0]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for group in groups:
                ng  = len(group)
                mgs = [generate_mask(B_, C, seq_len, r, device) for r in group]
                xa  = xf.repeat(ng, 1, 1) if ng > 1 else xf
                ma  = torch.cat(mgs, 0)   if ng > 1 else mgs[0]
                out = model(xa, mask=ma)
                mi  = 1.0 - ma
                lg  = (crit(out, xa) * mi).sum() / (mi.sum() + 1e-8) / n_passes
                scaler.scale(lg).backward()
                total += lg.item()
                del out, mi, lg, xa, ma, mgs
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(total)

            # Progress bar every 50 batches
            if (bi + 1) % 50 == 0 or (bi + 1) == n_batches:
                progress_bar(bi + 1, n_batches, np.mean(losses[-50:]), ep_t0)

        print()  # newline after progress bar
        avg_trn = np.mean(losses)
        sched.step()

        # ── Validate ──
        model.eval()
        vr = {r: [] for r in ratios}
        with torch.no_grad():
            for r in ratios:
                for vi, (bx, *_) in enumerate(val_dl):
                    bx = bx.float().to(device, non_blocking=True)
                    xf = bx.transpose(1, 2).contiguous()
                    m  = vmasks[r][vi].to(device)
                    out = model(xf, mask=m).float()
                    mi  = 1.0 - m
                    vr[r].append(
                        (crit(out, xf) * mi).sum().item() /
                        (mi.sum().item() + 1e-8))

        avg_val  = np.mean([np.mean(v) for v in vr.values()])
        ep_time  = (time.time() - ep_t0) / 60

        mark = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), ckpt)
            patience_cnt = 0
            mark = " ★"
        else:
            patience_cnt += 1

        vr_str = " | ".join(f"{r*100:.0f}%:{np.mean(vr[r]):.4f}" for r in ratios)
        log(f"Ep {ep+1:02d}/{t_cfg['epochs']} | "
            f"Train:{avg_trn:.4f} Val:{avg_val:.4f} | "
            f"{vr_str} | {ep_time:.1f}m{mark}")

        if patience_cnt >= t_cfg['patience']:
            log(f"Early stopping at epoch {ep+1}")
            break

    log(f"\nBest val MSE: {best_val:.6f} → {ckpt}")


if __name__ == '__main__':
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
