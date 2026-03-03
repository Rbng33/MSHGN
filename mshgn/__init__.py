from .model import MSHGN
from .data  import Dataset_ETT_hour

__version__ = "0.1.0"
__all__ = ["MSHGN", "Dataset_ETT_hour"]
```

---

## Résumé de la structure complète
```
MSHGN/
├── README.md              ← Résultats + architecture + quick start
├── requirements.txt
├── .gitignore
│
├── mshgn/
│   ├── __init__.py
│   ├── model.py           ← V1 (2.22M params, MSE=0.0387)
│   ├── model_v2.py        ← V2 (550K params, légère) [code du Cell 7]
│   └── data.py            ← Dataset_ETT_hour + Dataset_Jena [Cell 1-3+5]
│
├── configs/
│   ├── etth1_v1.yaml      ← Config V1 exacte de l'entraînement
│   └── etth1_v2.yaml      ← Config V2 légère
│
└── scripts/
    ├── train.py           ← Script propre reproductible
    └── test.py            ← Évaluation + per-channel + per-ratio
