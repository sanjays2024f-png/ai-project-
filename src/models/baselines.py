"""9 baselines, same splits + same SMOTE rule as Env-MeshNet.
Sklearn: LogReg, RF, XGBoost, SVM, MLP. Torch: 1D-CNN, LSTM, GRU, GAT-lite.
All predict AQI class; PM2.5 = class-median mapped in train.py for quick baseline.
STRICT: requires sklearn (+xgboost for xgb, +torch for cnn1d/lstm/gru/gat).
"""
from __future__ import annotations
import torch
import torch.nn as nn

HAS_TORCH = True

def sklearn_baseline(name: str, seed: int = 42):
    """Build sklearn/XGB classifier by name."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500, random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, eval_metric="mlogloss", random_state=seed)
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)
    raise ValueError(f"unknown sklearn baseline {name}")

SKLEARN_NAMES = ["logreg", "rf", "xgb", "svm", "mlp"]
TORCH_NAMES = ["cnn1d", "lstm", "gru", "gat"]
ALL_NAMES = ["envmeshnet"] + SKLEARN_NAMES + TORCH_NAMES  # 1 + 5 + 4 = 10 total

class SeqClf(torch.nn.Module):
    """Small torch baseline: cnn1d|lstm|gru|gat over feature vector."""
    def __init__(self, d: int, n_cls: int = 6, kind: str = "lstm", h: int = 64):
        super().__init__()
        self.kind = kind
        if kind == "cnn1d":
            self.net = torch.nn.Sequential(torch.nn.Conv1d(1, 16, 3, padding=1),
                torch.nn.ReLU(), torch.nn.AdaptiveAvgPool1d(8), torch.nn.Flatten(),
                torch.nn.Linear(16 * 8, h), torch.nn.ReLU())
        elif kind == "lstm":
            self.rnn = torch.nn.LSTM(d, h, batch_first=True)
            self.net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(h, h))
        elif kind == "gru":
            self.rnn = torch.nn.GRU(d, h, batch_first=True)
            self.net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(h, h))
        else:  # gat-lite: linear + self-attention over single token
            self.att = torch.nn.MultiheadAttention(h, 4, batch_first=True)
            self.proj = torch.nn.Linear(d, h)
            self.net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(h, h))
        self.head = torch.nn.Linear(h, n_cls)
    def forward(self, x):
        """x:(B,d) -> logits (B,n_cls)."""
        if self.kind == "cnn1d":
            h = self.net(x.unsqueeze(1))
        elif self.kind in ("lstm", "gru"):
            _, hh = self.rnn(x.unsqueeze(1))
            hh = hh[0] if self.kind == "lstm" else hh
            h = self.net(hh.squeeze(0))
        else:
            p = self.proj(x).unsqueeze(1)
            a, _ = self.att(p, p, p)
            h = self.net(a.squeeze(1))
        return self.head(h)

