# MSHGN: Multiscale Hypergraph Network for Multivariate Time Series Imputation


> Architecture: `LocalConv → MultiscaleMixing → TemporalHE → ChannelHE → H2N`

## Results (ETTh1, averaged over 12.5%/25%/37.5%/50% masking)

| Model | MSE | MAE | Parameters | Epochs |
|-------|-----|-----|-----------|--------|
| HGTS-Former (2025) | 0.085 | 0.192 | 10.38M | 20 |
| TimeMixer++ (2025) | 0.091 | 0.198 | — | — |
| **MSHGN V1 (ours)** | **0.0387** | **0.1301** | **2.22M** | 5 |

**−54.4% MSE, −78.6% parameters** vs HGTS-Former.  
Trained for only **5 epochs** (val loss still decreasing) — conservative estimate.

### Per-masking-rate breakdown

| Masking | MSE | MAE |
|---------|-----|-----|
| 12.5% | 0.0264 | 0.1106 |
| 25.0% | 0.0331 | 0.1217 |
| 37.5% | 0.0416 | 0.1353 |
| 50.0% | 0.0536 | 0.1527 |

## Architecture

MSHGN is purpose-built for imputation (not adapted from forecasting).  
Each block performs:
```
Input H
  │
  ├─► LocalTemporalConv        # depthwise-separable, k=9, within-channel
  │
  ├─► MultiscaleMixing         # U-Net season-trend, 4 scales, within-channel
  │
  ├─► TemporalHyperedge        # per-timestep cross-channel attention
  │
  ├─► ChannelHyperedge         # overlap-weighted structural cross-channel
  │
  └─► HyperedgeToNode          # rank-2 additive fusion
```

Three blocks are stacked (N=3) for progressive refinement.

### Key design principles

1. **Sequential separation** — temporal context built BEFORE cross-channel sharing
2. **Pervasive mask-awareness** — RevIN, embedding, attention, loss all handle missingness
3. **Point-level granularity** — no patch aggregation, each timestep independently addressable
4. **Dual hyperedge** — temporal (dynamic system state) + channel (structural relationships)



## Installation
```bash
git clone https://github.com/your-username/MSHGN.git
cd MSHGN
pip install -r requirements.txt
```

## Quick Start
```bash
# Train V1 (best results, requires ~14GB GPU)
python scripts/train.py --config configs/etth1_v1.yaml \
    --data_path /path/to/ETTh1.csv

# Train V2 (lightweight, ~3GB GPU)
python scripts/train.py --config configs/etth1_v2.yaml \
    --data_path /path/to/ETTh1.csv

# Evaluate
python scripts/test.py --config configs/etth1_v1.yaml \
    --checkpoint mshgn_best.pth \
    --data_path /path/to/ETTh1.csv
```

## Datasets

- **ETTh1**: 7 channels, hourly, 17,420 samples. [Download](https://github.com/zhouhaoyi/ETDataset)

Place CSV files in `data/` or pass `--data_path` directly.

## Model Variants

| Variant | d | Scales | Params | GPU | MSE |
|---------|---|--------|--------|-----|-----|
| V1 (best) | 128 | 3 | 2.22M | ~14GB | **0.0387** |
| V2 (light) | 64 | 2 | ~550K | ~3GB | TBD |

## Citation
```bibtex
@misc{mshgn2025,
  title   = {MSHGN: Multiscale Hypergraph Network for
             Multivariate Time Series Imputation},
  year    = {2025},
}
```

## References

- HGTS-Former: Wang et al., IEEE Transactions, 2025
- RevIN: Kim et al., ICLR 2022
- TimesNet: Wu et al., ICLR 2023
- U-Net: Ronneberger et al., MICCAI 2015
