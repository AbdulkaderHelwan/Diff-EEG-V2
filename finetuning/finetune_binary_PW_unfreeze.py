#!/usr/bin/env python3
"""
Fine-tune EEG Diffusion Model — ULTRA SIMPLE VERSION
Pure numpy loading, no lazy datasets, no classes, just straightforward code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torch.distributions import Bernoulli
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             confusion_matrix, classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path as _Path
# Ensure parent package path is importable (allows importing the v2 training module)
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel


BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all"
DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
NORM_STATS_DIR = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"

# Use v2 "large" config to match pretrained run_config.json
ARCH = dict(in_channels=22, model_channels=64, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=3, time_emb_dim=768, dropout=0.1, attention_heads=8)

# ---- runtime config (honor the args passed by the .sh launcher) ----
import argparse
import random
from torch.utils.data import Dataset, IterableDataset, get_worker_info

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--epochs", type=int, default=100)
_p.add_argument("--batch-size", type=int, default=256)
_p.add_argument("--val-batch-size", type=int, default=256)
_p.add_argument("--num-workers", type=int, default=4)
_p.add_argument("--lr", type=float, default=5e-4)
_p.add_argument("--shuffle-buffer", type=int, default=4000,
                help="per-worker sample shuffle buffer for the training stream")
_p.add_argument("--dryrun", action="store_true")
# ---- backbone fine-tuning controls (this is the "unfreeze" variant) ----
_p.add_argument("--unfreeze", choices=["none", "last_level", "last_two_levels",
                                       "encoder_all", "all"], default="last_two_levels",
                help="how much of the (in-path) backbone encoder to fine-tune")
_p.add_argument("--backbone-lr", type=float, default=None,
                help="LR for unfrozen backbone params (default: LR/10, discriminative)")
_p.add_argument("--grad-checkpoint", action="store_true",
                help="gradient-checkpoint encoder blocks to cut activation memory "
                     "(needed for larger unfreeze sets / batch sizes)")
_args, _ = _p.parse_known_args()

BATCH_SIZE = _args.batch_size
VAL_BATCH_SIZE = _args.val_batch_size
NUM_EPOCHS = _args.epochs
NUM_WORKERS = _args.num_workers
LR = _args.lr
BACKBONE_LR = _args.backbone_lr if _args.backbone_lr is not None else LR / 10.0
UNFREEZE = _args.unfreeze
GRAD_CHECKPOINT = _args.grad_checkpoint
SHUFFLE_BUFFER = _args.shuffle_buffer
PATIENCE = 20

# ================================================================
# DATA LOADING
# The full dataset is ~229 GB float32 — too large for RAM. Earlier we used a
# per-sample memory-mapped dataset, but on Lustre 1.86M *random* reads/epoch
# starved the GPU (~20 h/epoch, <7% CPU). Training now streams whole batch
# files SEQUENTIALLY (each .npy = ~2000 samples) with a shuffle buffer, turning
# ~1.86M random reads into ~980 sequential ones. Val/eval (smaller, read once
# per epoch) keep the simple memory-mapped path.
# ================================================================
SKIP_DATA_LOAD = _args.dryrun

if not SKIP_DATA_LOAD:
    print("Loading normalization stats...")
    mean = np.load(Path(NORM_STATS_DIR) / "mean.npy").astype(np.float32)
    std = np.load(Path(NORM_STATS_DIR) / "std.npy").astype(np.float32)
    print(f"  ✓ Loaded\n")
else:
    print("Dry-run mode: skipping normalization and data loading")
    mean = np.zeros((ARCH['in_channels'],), dtype=np.float32)
    std = np.ones((ARCH['in_channels'],), dtype=np.float32)


def gather_files(dir_label_pairs):
    """Return [(path, label), ...] for all batch .npy files in the given dirs."""
    out = []
    for directory, label in dir_label_pairs:
        for f in sorted(Path(directory).glob("*_batch_*.npy")):
            if "_labels" not in f.name:
                out.append((f, label))
    return out


class ClassInterleavedDataset(IterableDataset):
    """Sequential file reads, but classes interleaved at their natural ratio.

    The .npy files are single-class, so naively streaming them (even with a
    shuffle buffer) produces class-bursty batches — fatal for imbalanced data,
    where the minority class would appear in concentrated bursts instead of
    uniformly. Instead we keep ONE stream per class, read whole files
    sequentially (Lustre-friendly) into per-class shuffle buffers, and draw each
    sample from a class chosen in proportion to its remaining samples. Result:
    every batch is uniformly class-mixed (like true per-sample shuffling) while
    I/O stays sequential. RAM ~= buffer per class per worker.
    """

    def __init__(self, file_label_pairs, mean, std, buffer_size=4000, base_seed=42):
        self.mean = mean.reshape(-1, 1).astype(np.float32)   # (C, 1)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.buffer_size = max(2, buffer_size)
        self.base_seed = base_seed
        self.epoch = 0
        self.files_by_class = {}
        for f, l in file_label_pairs:
            self.files_by_class.setdefault(int(l), []).append(f)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        info = get_worker_info()
        rng = random.Random(self.base_seed + self.epoch)
        classes = sorted(self.files_by_class)

        files, ptr, bufs = {}, {}, {}
        for c in classes:
            fs = list(self.files_by_class[c])
            rng.shuffle(fs)
            if info is not None:                      # shard each class across workers
                fs = fs[info.id::info.num_workers]
            files[c], ptr[c], bufs[c] = fs, 0, []

        def refill(c):
            if ptr[c] < len(files[c]):
                arr = np.load(files[c][ptr[c]]).astype(np.float32)   # (n, C, T) sequential
                ptr[c] += 1
                arr = (arr - self.mean) / (self.std + 1e-6)
                order = list(range(len(arr)))
                rng.shuffle(order)
                bufs[c].extend(arr[i] for i in order)
                return True
            return False

        def remaining(c):  # samples still to emit (buffered + unread), ~2000/file
            return len(bufs[c]) + (len(files[c]) - ptr[c]) * 2000

        for c in classes:
            refill(c)

        active = [c for c in classes if remaining(c) > 0]
        while active:
            c = rng.choices(active, weights=[remaining(k) for k in active], k=1)[0]
            if not bufs[c] and not refill(c):
                active = [k for k in classes if remaining(k) > 0]
                continue
            x = bufs[c].pop(rng.randrange(len(bufs[c])))
            if len(bufs[c]) < self.buffer_size // 2:
                refill(c)
            yield torch.from_numpy(np.ascontiguousarray(x)), c
            active = [k for k in classes if remaining(k) > 0]


class LazyEEGDataset(Dataset):
    """Map-style per-sample memory-mapped dataset (used for val/eval)."""

    def __init__(self, file_label_pairs, mean, std):
        self.mean = mean.reshape(-1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.files = []
        self.labels_per_file = []
        self.index = []
        for f, label in file_label_pairs:
            n = np.load(f, mmap_mode='r').shape[0]
            fid = len(self.files)
            self.files.append(f)
            self.labels_per_file.append(label)
            self.index.extend((fid, i) for i in range(n))
        self.labels = np.array([self.labels_per_file[fid] for fid, _ in self.index], dtype=np.int64)
        self._cache = {}

    def _arr(self, fid):
        a = self._cache.get(fid)
        if a is None:
            a = np.load(self.files[fid], mmap_mode='r')
            self._cache[fid] = a
        return a

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, row = self.index[idx]
        x = np.asarray(self._arr(fid)[row], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(x), int(self.labels_per_file[fid])


print("="*60)
print("INDEXING DATA (sequential block-shuffle for train)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not SKIP_DATA_LOAD:
    # Finetune = train + dev pooled; label 0 = normal, 1 = seizure
    all_files = gather_files([
        (f"{BASE}/train-non-seizure", 0),
        (f"{BASE}/train-seizure",     1),
        (f"{BASE}/dev-non-seizure",   0),
        (f"{BASE}/dev-seizure",       1),
    ])
    # File-level stratified ~5% validation split (keeps val files out of train)
    split_rng = random.Random(42)
    by_class = {0: [], 1: []}
    for f, l in all_files:
        by_class[l].append((f, l))
    train_files, val_files = [], []
    for l, items in by_class.items():
        split_rng.shuffle(items)
        k = max(1, int(round(0.05 * len(items))))
        val_files += items[:k]
        train_files += items[k:]

    # Count samples per file (header reads only) for class weights + LR schedule
    n0 = n1 = 0
    for f, l in tqdm(train_files, desc="  Counting train samples"):
        n = np.load(f, mmap_mode='r').shape[0]
        if l == 0:
            n0 += n
        else:
            n1 += n
    n_train = n0 + n1
    STEPS_PER_EPOCH = max(1, n_train // BATCH_SIZE)

    train_ds = ClassInterleavedDataset(train_files, mean, std, buffer_size=SHUFFLE_BUFFER)
    val_ds = LazyEEGDataset(val_files, mean, std)
    eval_ds = LazyEEGDataset(gather_files([
        (f"{BASE}/eval-non-seizure", 0),
        (f"{BASE}/eval-seizure",     1),
    ]), mean, std)

    w_n = n_train / (2.0 * max(n0, 1))
    w_s = n_train / (2.0 * max(n1, 1))
    print(f"\n✓ Train files {len(train_files)} | {n_train:,} samples "
          f"(normal {n0:,} / seizure {n1:,})")
    print(f"✓ Val   files {len(val_files)} | {len(val_ds):,} samples")
    print(f"✓ Eval  | {len(eval_ds):,} samples")
    print(f"✓ Steps/epoch {STEPS_PER_EPOCH:,} | class weights normal={w_n:.3f} seizure={w_s:.3f}\n")
else:
    print("Creating small dummy datasets for dry-run")
    n = 16
    Xd = torch.randn(2 * n, ARCH['in_channels'], 1280)
    yd = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    train_ds = TensorDataset(Xd, yd)
    val_ds = TensorDataset(Xd[:8], yd[:8])
    eval_ds = TensorDataset(Xd[:8], yd[:8])
    w_n = w_s = 1.0
    STEPS_PER_EPOCH = max(1, (2 * n) // BATCH_SIZE)

print(f"Using device: {device}")

# Data loaders. Train is an IterableDataset (no shuffle arg; shuffling is
# internal). persistent_workers=False so set_epoch() reshuffles each epoch.
_eval_kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
if NUM_WORKERS > 0:
    _eval_kw["persistent_workers"] = True
    _eval_kw["prefetch_factor"] = 4
_train_kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
if NUM_WORKERS > 0:
    _train_kw["prefetch_factor"] = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True, **_train_kw)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, **_eval_kw)
eval_loader = DataLoader(eval_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, **_eval_kw)

# ================================================================
# MODELS
# ================================================================
class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, num_classes=2, dropout=0.4,
                 grad_checkpoint=False):
        super().__init__()
        self.backbone = backbone
        self.grad_checkpoint = grad_checkpoint
        self.register_buffer('probe_timesteps', torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        multi_scale_per_t = backbone.model_channels * sum(backbone.channel_multipliers)
        aggregated_dim = multi_scale_per_t * len(self.probe_timesteps)
        
        self.classifier = nn.Sequential(
            nn.Linear(aggregated_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, num_classes),
        )

    def _run_block(self, block, h, t_emb, takes_time):
        # Optionally gradient-checkpoint to trade compute for activation memory.
        # Only worthwhile while training, with grad enabled, AND when the block has
        # trainable params (checkpointing a fully-frozen block just wastes a recompute).
        trainable = any(p.requires_grad for p in block.parameters())
        if self.grad_checkpoint and trainable and self.training and torch.is_grad_enabled():
            if takes_time:
                return checkpoint(block, h, t_emb, use_reentrant=False)
            return checkpoint(block, h, use_reentrant=False)
        return block(h, t_emb) if takes_time else block(h)

    def _encode_at_t(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        level_outputs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = self._run_block(module_list[0], h, t_emb, takes_time=False)
            else:
                for block in module_list:
                    takes_time = (hasattr(block, 'forward')
                                  and 'time_emb' in block.forward.__code__.co_varnames)
                    h = self._run_block(block, h, t_emb, takes_time)
                level_outputs.append(self.pool(h).squeeze(-1))
        return torch.cat(level_outputs, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = [self._encode_at_t(x, ts.expand(B)) for ts in self.probe_timesteps]
        combined = torch.cat(feats, dim=1)
        logits = self.classifier(combined)
        return logits, combined


class ReinforcedDecisionLayer(nn.Module):
    def __init__(self, input_dim, rl_weight=0.1, momentum=0.9):
        super().__init__()
        self.rl_weight = rl_weight
        self.momentum = momentum
        hidden = max(64, input_dim // 4)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, logits, features, training=False):
        adj = self.policy(features).squeeze(-1)
        adjusted = logits.clone()
        adjusted[:, 1] += adj
        adjusted[:, 0] -= adj
        probs = torch.softmax(adjusted, dim=1)[:, 1]
        
        if training:
            probs = probs.clamp(1e-6, 1 - 1e-6)
            dist = Bernoulli(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return adjusted, probs, actions, log_probs
        return adjusted, probs, None, None

    def update_baseline(self, reward):
        self.baseline.mul_(self.momentum).add_((1 - self.momentum) * reward.detach())
        return self.baseline.detach()


def batch_f1(preds, labels, eps=1e-6):
    p, l = preds.float(), labels.float()
    tp = (p * l).sum()
    fp = (p * (1 - l)).sum()
    fn = ((1 - p) * l).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)


def _autocast(device):
    # bf16 mixed precision on CUDA (no GradScaler needed for bf16); no-op on CPU
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_epoch(classifier, rl_layer, loader, optimizer, scheduler, loss_fn, device):
    classifier.train()
    rl_layer.train()
    classifier.backbone.eval()

    total_loss, preds_all, labels_all, rewards = [], [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        with _autocast(device):
            logits, features = classifier(x)
            adj, probs, acts, lp = rl_layer(logits, features, training=True)
            ce = loss_fn(adj, y)
        reward = batch_f1(acts, y)
        baseline = rl_layer.update_baseline(reward)
        rl_loss = -(reward.detach() - baseline) * lp.mean()
        loss = ce + rl_layer.rl_weight * rl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.classifier.parameters()) + list(rl_layer.parameters())
            + trainable_backbone, 1.0)
        optimizer.step()
        if scheduler:
            try:
                scheduler.step()
            except ValueError:
                pass  # OneCycle total steps reached (per-worker drop_last rounding)

        total_loss.append(loss.item())
        rewards.append(reward.item())
        with torch.no_grad():
            preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    
    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    return float(np.mean(total_loss)), acc, float(np.mean(rewards))


@torch.no_grad()
def evaluate(classifier, rl_layer, loader, loss_fn, device):
    classifier.eval()
    rl_layer.eval()
    total_loss, probs_all, preds_all, labels_all = 0.0, [], [], []
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device):
            logits, features = classifier(x)
            adj, probs, _, _ = rl_layer(logits, features, training=False)
            total_loss += loss_fn(adj, y).item()
        probs_all.extend(probs.float().cpu().numpy())
        preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    
    labels = np.array(labels_all)
    probs = np.array(probs_all)
    acc = np.mean(np.array(preds_all) == labels)
    
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, probs)
        prec_c, rec_c, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rec_c, prec_c)
    else:
        roc_auc = pr_auc = 0.5
    
    return total_loss / len(loader), acc, roc_auc, pr_auc, probs, labels


# ================================================================
# MAIN TRAINING
# ================================================================
print("\nLoading backbone...")
backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
try:
    ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
    backbone.load_state_dict(ckpt['model_state_dict'])
    print("  ✓ Checkpoint loaded into backbone")
except Exception as e:
    print(f"  ! Warning: could not load checkpoint ({e}). Proceeding without state_dict load.")
backbone.eval()

# ----------------------------------------------------------------------
# SELECTIVE UNFREEZE (real fine-tuning).
# Only modules on the classifier's forward path are worth unfreezing:
# init_conv, the model-level time_mlp, and down_blocks (the encoder).
# bottleneck_blocks / up_blocks are never executed by EEGClassifier, so
# unfreezing them would only waste memory. We unfreeze whole encoder levels
# from the deepest end (closest to the classifier head), including the
# downsample convs that sit between unfrozen levels.
# ----------------------------------------------------------------------
for p in backbone.parameters():
    p.requires_grad = False


def _unfreeze_module(m):
    for p in m.parameters():
        p.requires_grad = True


def _is_level_group(module_list):
    # A "level" is a ModuleList of residual blocks; a downsample is a lone Conv1d.
    return not (len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d))


if UNFREEZE == "none":
    pass
elif UNFREEZE == "all":
    _unfreeze_module(backbone.init_conv)
    _unfreeze_module(backbone.time_mlp)
    for module_list in backbone.down_blocks:
        _unfreeze_module(module_list)
elif UNFREEZE == "encoder_all":
    for module_list in backbone.down_blocks:
        _unfreeze_module(module_list)
else:
    # last_level -> 1 deepest level, last_two_levels -> 2 deepest levels.
    n_levels = 1 if UNFREEZE == "last_level" else 2
    seen = 0
    for module_list in reversed(backbone.down_blocks):
        _unfreeze_module(module_list)          # unfreeze groups AND interleaved downsamples
        if _is_level_group(module_list):
            seen += 1
            if seen >= n_levels:
                break

trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
n_bb = sum(p.numel() for p in trainable_backbone)
print(f"  ✓ Backbone prepared | unfreeze='{UNFREEZE}' | "
      f"trainable backbone params: {n_bb:,} ({len(trainable_backbone)} tensors)\n")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# Everything for this experiment lives under EEGdiff_V2/Binary_finetune_unfrozen/
# (one run subfolder per launch holds the model checkpoint, history, AND results).
OUT_ROOT = Path("/home/abdulh/scratch/EEGdiff_V2/Binary_finetune_unfrozen")
output_path = OUT_ROOT / f"run_{UNFREEZE}_{ts}"
output_path.mkdir(parents=True, exist_ok=True)

# Build models AFTER backbone is ready so we compute dims from instantiated backbone
multi_scale_dim = getattr(backbone, 'model_channels', ARCH['model_channels']) * sum(getattr(backbone, 'channel_multipliers', ARCH['channel_multipliers']))
agg_dim = multi_scale_dim * 5
classifier = EEGClassifier(backbone, feature_dim=256, dropout=0.4,
                           grad_checkpoint=GRAD_CHECKPOINT).to(device)

rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=0.1).to(device)

# Param groups: classifier head + RL policy at LR; unfrozen backbone at a smaller
# discriminative LR (BACKBONE_LR). The backbone group is only added if non-empty.
param_groups = [
    {'params': list(classifier.classifier.parameters()), 'lr': LR, 'weight_decay': 1e-4},
    {'params': list(rl_layer.parameters()), 'lr': LR, 'weight_decay': 1e-4},
]
max_lrs = [LR, LR]
if trainable_backbone:
    param_groups.append({'params': trainable_backbone, 'lr': BACKBONE_LR, 'weight_decay': 1e-4})
    max_lrs.append(BACKBONE_LR)
    print(f"  ✓ Optimizer: head/RL LR={LR:.2e}, backbone LR={BACKBONE_LR:.2e}")
else:
    print(f"  ✓ Optimizer: head/RL LR={LR:.2e}, backbone frozen")

optimizer = optim.AdamW(param_groups)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lrs, epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH, pct_start=0.1, anneal_strategy='cos',
)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([w_n, w_s], dtype=torch.float32).to(device))

if SKIP_DATA_LOAD:
    # Quick dry-run: create a tiny batch, run forward through classifier + rl layer
    print("\n=== Dry-run forward test ===")
    xb = torch.randn(4, ARCH['in_channels'], 1280, device=device)
    yb = torch.tensor([0,1,0,1], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, feats = classifier(xb)
        adj, probs, acts, lp = rl_layer(logits, feats, training=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Adjusted logits shape: {adj.shape}")
    print(f"Probs shape: {probs.shape}")
    # sample loss
    ce = loss_fn(adj, yb)
    print(f"Sample CE loss: {ce.item():.6f}")
    print("Dry-run complete. Exiting.")
    # mark dummy task done in todo list via file write (user can run full training afterwards)
    sys.exit(0)

# Training
print("="*60)
print("TRAINING")
print("="*60 + "\n")

best_dev_auc = 0.0
patience_counter = 0
history = {k: [] for k in ['train_loss','train_acc','train_f1','dev_loss','dev_acc','dev_auc','dev_pr_auc']}


for epoch in range(NUM_EPOCHS):
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)  # reshuffle the streaming order
    tr_loss, tr_acc, tr_f1 = train_epoch(classifier, rl_layer, train_loader, optimizer, scheduler, loss_fn, device)
    val_loss, val_acc, val_auc, val_pr, _, _ = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['train_f1'].append(tr_f1)
    history['dev_loss'].append(val_loss)
    history['dev_acc'].append(val_acc)
    history['dev_auc'].append(val_auc)
    history['dev_pr_auc'].append(val_pr)
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d} | Train Loss:{tr_loss:.4f} Acc:{tr_acc:.3f} F1:{tr_f1:.3f} | "
          f"Val Loss:{val_loss:.4f} Acc:{val_acc:.3f} ROC:{val_auc:.3f} PR:{val_pr:.3f} | LR:{lr:.2e}",
          flush=True)
    if val_auc > best_dev_auc:
        best_dev_auc = val_auc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'classifier': classifier.state_dict(),
            'rl_layer': rl_layer.state_dict(),
            'best_dev_auc': best_dev_auc,
            'arch': ARCH,
            'unfreeze': UNFREEZE,
            'backbone_lr': BACKBONE_LR,
        }, output_path / "best_classifier.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Load best and evaluate
best = torch.load(output_path / "best_classifier.pth", map_location=device)
classifier.load_state_dict(best['classifier'])
rl_layer.load_state_dict(best['rl_layer'])


_, _, _, _, val_probs, val_labels = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
_, _, _, _, eval_probs, eval_labels = evaluate(classifier, rl_layer, eval_loader, loss_fn, device)

# Results live in the SAME run folder as the model + history (everything together).
res_dir = output_path
np.savez_compressed(res_dir / "scores.npz",
                    val_probs=val_probs, val_labels=val_labels,
                    eval_probs=eval_probs, eval_labels=eval_labels)

# F1-optimal threshold chosen on val (no eval leakage), then applied to eval.
from sklearn.metrics import f1_score
_pc, _rc, _thr = precision_recall_curve(val_labels, val_probs)
_f1 = 2 * _pc[:-1] * _rc[:-1] / (_pc[:-1] + _rc[:-1] + 1e-12)
thr_star = float(_thr[int(np.nanargmax(_f1))])

results = {"unfreeze": UNFREEZE, "backbone_lr": BACKBONE_LR, "arch": ARCH,
           "epoch": int(best.get("epoch")) if best.get("epoch") is not None else None,
           "best_dev_auc": float(best.get("best_dev_auc")) if best.get("best_dev_auc") is not None else None,
           "f1_optimal_threshold": thr_star, "splits": {}}
report_lines = [f"unfreeze: {UNFREEZE}   backbone_lr: {BACKBONE_LR:.2e}",
                f"epoch: {results['epoch']}   best_dev_auc: {results['best_dev_auc']}",
                f"F1-optimal threshold (from val): {thr_star:.4f}\n"]

for name, probs, labels in [("val", val_probs, val_labels), ("eval", eval_probs, eval_labels)]:
    roc = float(roc_auc_score(labels, probs))
    pc, rc, _ = precision_recall_curve(labels, probs)
    pr_auc = float(auc(rc, pc))
    preds_05 = (probs >= 0.5).astype(int)
    preds_t = (probs >= thr_star).astype(int)
    f1_05 = float(f1_score(labels, preds_05, pos_label=1))
    f1_t = float(f1_score(labels, preds_t, pos_label=1))
    rep_05 = classification_report(labels, preds_05, target_names=['Normal', 'Seizure'], digits=4)
    rep_t = classification_report(labels, preds_t, target_names=['Normal', 'Seizure'], digits=4)

    block = (f"\n{'='*60}\n{name.upper()} RESULTS\n{'='*60}\n"
             f"ROC-AUC: {roc:.4f}    PR-AUC: {pr_auc:.4f}\n"
             f"\n-- threshold = 0.50 --\nseizure F1: {f1_05:.4f}\n{rep_05}\n"
             f"-- threshold = {thr_star:.4f} (F1-optimal, from val) --\nseizure F1: {f1_t:.4f}\n{rep_t}")
    print(block)
    report_lines.append(block)
    results["splits"][name] = {"roc_auc": roc, "pr_auc": pr_auc,
                               "thr_0.5": {"seizure_f1": f1_05, "report": rep_05},
                               "thr_optimal": {"seizure_f1": f1_t, "report": rep_t}}

(res_dir / "results.json").write_text(json.dumps(results, indent=2))
(res_dir / "results.txt").write_text("\n".join(report_lines) + "\n")

with open(output_path / "history.json", 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print(f"\n✓ All outputs (model + history + results) saved to {output_path}/")

