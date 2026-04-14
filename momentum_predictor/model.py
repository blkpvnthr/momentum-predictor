from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# DEVICE
# =========================================================

def get_device(preferred: str | torch.device | None = None) -> torch.device:
    if isinstance(preferred, torch.device):
        return preferred

    if preferred is not None:
        preferred = preferred.lower()

        if preferred == "cpu":
            return torch.device("cpu")

        if preferred == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("CUDA requested but not available.")

        if preferred == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("MPS requested but not available.")

        if preferred == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("No GPU backend available (CUDA or MPS).")

        raise ValueError(f"Invalid preferred device: {preferred}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No GPU backend available (CUDA or MPS).")

# =========================================================
# TARGET SCALER
# =========================================================
@dataclass
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y_train: np.ndarray, n_return_targets: int = 3) -> "TargetScaler":
        mean = np.zeros(y_train.shape[1], dtype=np.float32)
        std = np.ones(y_train.shape[1], dtype=np.float32)

        ret_mean = y_train[:, :n_return_targets].mean(axis=0)
        ret_std = y_train[:, :n_return_targets].std(axis=0) + 1e-6

        mean[:n_return_targets] = ret_mean
        std[:n_return_targets] = ret_std
        return cls(mean=mean, std=std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform_returns(self, y_ret: np.ndarray) -> np.ndarray:
        return y_ret * self.std[: y_ret.shape[1]] + self.mean[: y_ret.shape[1]]

    def inverse_transform_return_index(self, y_ret_1d: np.ndarray, idx: int) -> np.ndarray:
        return y_ret_1d * self.std[idx] + self.mean[idx]


# =========================================================
# BUILDING BLOCKS
# =========================================================
class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, g = self.proj(x).chunk(2, dim=-1)
        x = a * torch.sigmoid(g)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.15):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# =========================================================
# MODELS
# =========================================================
class DualStreamTransformer(nn.Module):
    def __init__(
        self,
        input_dim_1m: int,
        input_dim_5m: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        max_len: int = 256,
    ):
        super().__init__()

        self.proj_1m = nn.Linear(input_dim_1m, d_model)
        self.proj_5m = nn.Linear(input_dim_5m, d_model)

        self.cls_1m = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_5m = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_1m = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.pos_5m = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        enc1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        enc5 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder_1m = nn.TransformerEncoder(enc1, n_layers)
        self.encoder_5m = nn.TransformerEncoder(enc5, n_layers)

        self.fusion = nn.Sequential(
            GLUBlock(d_model * 2, d_model, dropout=dropout),
            ResidualMLPBlock(d_model, hidden_mult=2, dropout=dropout),
            ResidualMLPBlock(d_model, hidden_mult=2, dropout=dropout),
        )

        self.return_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.return_bias_head = nn.Linear(d_model, 1)

        self.breakout_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.cont_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode(self, x, proj, cls, pos, encoder):
        b = x.size(0)
        x = proj(x)
        cls_tok = cls.expand(b, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = x + pos[:, : x.size(1)]
        x = encoder(x)
        return x[:, 0]

    def forward(self, x1, x5):
        x1 = self.encode(x1, self.proj_1m, self.cls_1m, self.pos_1m, self.encoder_1m)
        x5 = self.encode(x5, self.proj_5m, self.cls_5m, self.pos_5m, self.encoder_5m)

        x = torch.cat([x1, x5], dim=1)
        x = self.fusion(x)

        raw_returns = self.return_head(x)
        return_bias = self.return_bias_head(x)
        returns = raw_returns + return_bias

        return {
            "returns": returns,
            "breakout": self.breakout_head(x),
            "continuation": self.cont_head(x),
        }


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 160, dropout: float = 0.20):
        super().__init__()

        self.input_proj = GLUBlock(input_dim, hidden_dim, dropout=dropout)
        self.backbone = nn.Sequential(
            ResidualMLPBlock(hidden_dim, hidden_mult=2, dropout=dropout),
            ResidualMLPBlock(hidden_dim, hidden_mult=2, dropout=dropout),
            ResidualMLPBlock(hidden_dim, hidden_mult=2, dropout=dropout),
        )

        self.return_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.return_bias_head = nn.Linear(hidden_dim, 1)

        self.breakout_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.cont_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)

        raw_returns = self.return_head(x)
        return_bias = self.return_bias_head(x)
        returns = raw_returns + return_bias

        return {
            "returns": returns,
            "breakout": self.breakout_head(x),
            "continuation": self.cont_head(x),
        }


# =========================================================
# TCN
# =========================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.15):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)

        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.norm2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)


class TCNBaseline(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()

        layers = []
        channels = [64, 96, 128]

        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2**i, dropout=dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels[-1])

        self.return_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 3),
        )
        self.return_bias_head = nn.Linear(channels[-1], 1)

        self.breakout_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, 2),
        )
        self.cont_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.norm(x)

        raw_returns = self.return_head(x)
        return_bias = self.return_bias_head(x)
        returns = raw_returns + return_bias

        return {
            "returns": returns,
            "breakout": self.breakout_head(x),
            "continuation": self.cont_head(x),
        }


# =========================================================
# HELPERS
# =========================================================
def normalize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train - mean) / std, (test - mean) / std


def to_tensor(x, device: torch.device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def flatten(x: np.ndarray):
    return x.reshape(x.shape[0], -1)


# =========================================================
# LOSS
# =========================================================
def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """
    Pairwise ranking loss on the 15m horizon.
    Encourages stronger separation between better and worse outcomes.
    """
    n = pred.size(0)
    if n < 2:
        return pred.new_tensor(0.0)

    idx_i = torch.arange(n, device=pred.device)
    idx_j = torch.roll(idx_i, shifts=1)

    pred_diff = pred[idx_i] - pred[idx_j]
    target_diff = target[idx_i] - target[idx_j]

    sign = torch.sign(target_diff)
    valid = sign != 0

    if valid.sum() == 0:
        return pred.new_tensor(0.0)

    signed_margin = sign[valid] * pred_diff[valid]
    return F.softplus(-(signed_margin - margin)).mean()


def directional_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary directional loss for 15m return.
    """
    target_up = (target > 0).float()
    return F.binary_cross_entropy_with_logits(pred, target_up)


def compute_loss(
    out,
    y,
    tail_weight_alpha: float = 3.0,
    std_floor: float = 0.75,
    mean_align_weight: float = 0.05,
    rank_weight: float = 0.45,
    std_penalty_weight: float = 0.35,
    direction_weight: float = 0.35,
):
    r = y[:, :3]
    b = y[:, 3:5]
    c = y[:, 5:6]

    pred_r = out["returns"]

    # emphasize larger realized moves
    tail_weight = 1.0 + tail_weight_alpha * torch.mean(torch.abs(r), dim=1, keepdim=True)

    return_err = F.smooth_l1_loss(pred_r, r, reduction="none")
    lr = (return_err * tail_weight).mean()

    # heavier focus on 15m horizon
    horizon_weights = torch.tensor([0.35, 2.50, 0.35], device=r.device).view(1, 3)
    lr_h = (return_err * horizon_weights * tail_weight).mean()

    # align batch mean slightly, but not too strongly
    pred_mean = pred_r.mean(dim=0)
    true_mean = r.mean(dim=0)
    mean_loss = F.smooth_l1_loss(pred_mean, true_mean)

    # force broader 15m prediction spread
    pred_std_15 = torch.std(pred_r[:, 1], unbiased=False)
    std_penalty = F.relu(std_floor - pred_std_15)

    # stronger cross-sectional ranking on 15m
    rank_loss = pairwise_rank_loss(pred_r[:, 1], r[:, 1], margin=0.05)

    # direct directional supervision on 15m
    direction_loss = directional_bce_loss(pred_r[:, 1], r[:, 1])

    lb = F.binary_cross_entropy_with_logits(out["breakout"], b)
    lc = F.binary_cross_entropy_with_logits(out["continuation"], c)

    total = (
        0.35 * lr
        + 1.10 * lr_h
        + rank_weight * rank_loss
        + direction_weight * direction_loss
        + std_penalty_weight * std_penalty
        + mean_align_weight * mean_loss
        + 0.18 * lb
        + 0.08 * lc
    )
    return total




# =========================================================
# METRICS
# =========================================================
def metrics(pred: torch.Tensor, true: torch.Tensor):
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()

    pred_std = float(np.std(p))
    true_std = float(np.std(t))

    if len(p) < 2 or pred_std < 1e-12 or true_std < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    top10 = p >= np.percentile(p, 90)
    bot10 = p <= np.percentile(p, 10)

    top20 = p >= np.percentile(p, 80)
    bot20 = p <= np.percentile(p, 20)

    top_mean = float(t[top10].mean()) if top10.any() else 0.0
    bot_mean = float(t[bot10].mean()) if bot10.any() else 0.0
    spread = float(top_mean - bot_mean) if top10.any() and bot10.any() else 0.0

    top20_mean = float(t[top20].mean()) if top20.any() else 0.0
    bot20_mean = float(t[bot20].mean()) if bot20.any() else 0.0
    spread20 = float(top20_mean - bot20_mean) if top20.any() and bot20.any() else 0.0

    hit = float((t[top10] > 0).mean()) if top10.any() else 0.0
    hit20 = float((t[top20] > 0).mean()) if top20.any() else 0.0

    mae = float(np.mean(np.abs(p - t))) if len(p) else 0.0
    rmse = float(np.sqrt(np.mean((p - t) ** 2))) if len(p) else 0.0

    return {
        "corr": corr,
        "spread": spread,
        "spread20": spread20,
        "hit": hit,
        "hit20": hit20,
        "top_mean": top_mean,
        "bot_mean": bot_mean,
        "top20_mean": top20_mean,
        "bot20_mean": bot20_mean,
        "pred_std": pred_std,
        "mae": mae,
        "rmse": rmse,
    }


# =========================================================
# WALK-FORWARD ENGINE
# =========================================================
def walk_forward(
    model_fn,
    X1,
    y,
    X5=None,
    is_tabular: bool = False,
    device: str = "cpu",
    train_window: int = 2500,
    test_window: int = 500,
    epochs: int = 18,
    lr: float = 2e-4,
    batch_size: int = 192,
    weight_decay: float = 2e-4,
):
    device = get_device(device)
    print(f"[device] {device}")

    results = []
    n = len(X1)

    if n < train_window + test_window:
        print(
            f"[walk_forward] not enough samples: "
            f"n={n}, train_window={train_window}, test_window={test_window}"
        )
        return None

    last_model = None

    for w, start in enumerate(range(0, n - train_window - test_window + 1, test_window)):
        print(f"\n=== WINDOW {w} ===")

        tr = slice(start, start + train_window)
        te = slice(start + train_window, start + train_window + test_window)

        X1_tr_raw, X1_te_raw = X1[tr], X1[te]
        y_tr_raw, y_te_raw = y[tr].copy(), y[te].copy()

        if is_tabular:
            mean = X1_tr_raw.mean(axis=0, keepdims=True)
            std = X1_tr_raw.std(axis=0, keepdims=True) + 1e-6
            X1_tr_np = (X1_tr_raw - mean) / std
            X1_te_np = (X1_te_raw - mean) / std
        else:
            X1_tr_np, X1_te_np = normalize(X1_tr_raw, X1_te_raw)

        y_scaler = TargetScaler.fit(y_tr_raw, n_return_targets=3)
        y_tr_scaled = y_scaler.transform(y_tr_raw)
        y_te_scaled = y_scaler.transform(y_te_raw)

        X1_tr = to_tensor(X1_tr_np, device)
        X1_te = to_tensor(X1_te_np, device)
        y_tr = to_tensor(y_tr_scaled, device)
        y_te = to_tensor(y_te_scaled, device)

        if X5 is not None:
            X5_tr_raw, X5_te_raw = X5[tr], X5[te]
            X5_tr_np, X5_te_np = normalize(X5_tr_raw, X5_te_raw)
            X5_tr = to_tensor(X5_tr_np, device)
            X5_te = to_tensor(X5_te_np, device)
        else:
            X5_tr = None
            X5_te = None

        model = model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

        best_metrics = None
        best_key = (
            float("-inf"),  # spread
            float("-inf"),  # spread20
            float("-inf"),  # top_mean
            float("-inf"),  # corr
            float("-inf"),  # pred_std
        )
        best_state = None

        for e in range(epochs):
            model.train()

            perm = torch.randperm(X1_tr.size(0), device=device)
            total_train_loss = 0.0
            n_batches = 0

            for i in range(0, X1_tr.size(0), batch_size):
                idx = perm[i : i + batch_size]

                xb1 = X1_tr[idx]
                yb = y_tr[idx]

                if X5_tr is not None:
                    xb5 = X5_tr[idx]
                    out = model(xb1, xb5)
                else:
                    out = model(xb1)

                loss = compute_loss(out, yb)

                if not torch.isfinite(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_train_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            train_loss = total_train_loss / max(n_batches, 1)

            model.eval()
            with torch.no_grad():
                if X5_te is not None:
                    out = model(X1_te, X5_te)
                else:
                    out = model(X1_te)

                test_loss = compute_loss(out, y_te)

                pred_scaled = out["returns"][:, 1].detach().cpu().numpy()
                true_scaled = y_te[:, 1].detach().cpu().numpy()

                pred_unscaled = y_scaler.inverse_transform_return_index(pred_scaled, idx=1)
                true_unscaled = y_scaler.inverse_transform_return_index(true_scaled, idx=1)

                pred_t = torch.tensor(pred_unscaled, dtype=torch.float32)
                true_t = torch.tensor(true_unscaled, dtype=torch.float32)

                m = metrics(pred_t, true_t)

            current_key = (
                m["spread"],
                m["spread20"],
                m["top_mean"],
                m["corr"],
                m["pred_std"],
            )
            if current_key > best_key:
                best_key = current_key
                best_metrics = {
                    "window": w,
                    "epoch": e + 1,
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss.item()),
                    **m,
                }
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"[W{w} E{e+1}] "
                f"train={train_loss:.4f} "
                f"test={test_loss.item():.4f} "
                f"corr={m['corr']:.4f} "
                f"spread10={m['spread']:.5f} "
                f"spread20={m['spread20']:.5f} "
                f"hit10={m['hit']:.2f} "
                f"hit20={m['hit20']:.2f} "
                f"top10={m['top_mean']:.6f} "
                f"pred_std={m['pred_std']:.6f}"
            )

        if best_metrics is None:
            best_metrics = {
                "window": w,
                "epoch": None,
                "train_loss": float("nan"),
                "test_loss": float("nan"),
                "corr": 0.0,
                "spread": 0.0,
                "hit": 0.0,
                "top_mean": 0.0,
                "bot_mean": 0.0,
                "pred_std": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
            }
        else:
            if best_state is not None:
                model.load_state_dict(best_state)

        print(
            f"[W{w} BEST] "
            f"epoch={best_metrics['epoch']} "
            f"train={best_metrics['train_loss']:.4f} "
            f"test={best_metrics['test_loss']:.4f} "
            f"corr={best_metrics['corr']:.4f} "
            f"spread10={best_metrics['spread']:.5f} "
            f"spread20={best_metrics['spread20']:.5f} "
            f"hit10={best_metrics['hit']:.2f} "
            f"hit20={best_metrics['hit20']:.2f} "
            f"top10={best_metrics['top_mean']:.6f} "
            f"bot10={best_metrics['bot_mean']:.6f} "
            f"pred_std={best_metrics['pred_std']:.6f}"
        )

        results.append(best_metrics)
        last_model = model

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)

    if results:
        print("\n=== AVERAGES ===")
        print(f"corr={np.mean([r['corr'] for r in results]):.4f}")
        print(f"spread={np.mean([r['spread'] for r in results]):.6f}")
        print(f"hit={np.mean([r['hit'] for r in results]):.3f}")
        print(f"top_mean={np.mean([r['top_mean'] for r in results]):.6f}")
        print(f"bot_mean={np.mean([r['bot_mean'] for r in results]):.6f}")
        print(f"pred_std={np.mean([r['pred_std'] for r in results]):.6f}")
        print(f"mae={np.mean([r['mae'] for r in results]):.6f}")
        print(f"rmse={np.mean([r['rmse'] for r in results]):.6f}")

    return last_model


# =========================================================
# ENTRY POINTS
# =========================================================
def train_tabular_baseline(X1, y, device: str = "cpu"):
    X = flatten(X1)
    return walk_forward(
        model_fn=lambda: TabularMLP(X.shape[-1]),
        X1=X,
        y=y,
        is_tabular=True,
        device=device,
    )


def train_tcn_baseline(X1, y, device: str = "cpu"):
    return walk_forward(
        model_fn=lambda: TCNBaseline(X1.shape[-1]),
        X1=X1,
        y=y,
        device=device,
    )


def train_model(X1, X5, y, device: str = "auto"):
    return walk_forward(
        model_fn=lambda: DualStreamTransformer(
            input_dim_1m=X1.shape[-1],
            input_dim_5m=X5.shape[-1],
        ),
        X1=X1,
        y=y,
        X5=X5,
        device=device,
    )