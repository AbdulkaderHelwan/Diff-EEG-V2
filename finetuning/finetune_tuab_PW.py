#!/usr/bin/env python3
"""
Fine-tune the EEG Diffusion backbone on TUAB (normal vs abnormal), patient-wise.

- Uses the pretrained EEGdiff_V2 "large" backbone as a frozen/partly-frozen
  feature extractor (same backbone + RL decision layer as finetune_binary_PW.py).
- Lazy, memory-mapped data loading (the full TUAB set is hundreds of GB).
- Runs a sequence of PROGRESSIVE UNFREEZING experiments: head-only first, then
  progressively unfreeze more of the encoder, and reports the best.
- Trains on TUAB `train-*`, tests on TUAB `test-*` (disjoint patients).
- Metrics: F2, F1, Accuracy, Precision, Recall (sensitivity), Specificity,
  ROC-AUC, PR-AUC.

Label convention: 0 = normal, 1 = abnormal (TUAB "abnormal" was processed into
the `*-seizure` folders).
"""

import argparse
import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path as _Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              TensorDataset, get_worker_info)
from tqdm import tqdm
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Make the v2 training module importable (for the backbone definition)
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel

# ================================================================
# CONFIG
# ================================================================
BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/processed_tuab"
DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
NORM_STATS_DIR = f"{BASE}/normalization"

# All TUAB finetuning outputs (checkpoints, results, summaries) go here.
OUT_ROOT = "/home/abdulh/scratch/EEGdiff_V2/TUAB_Res"

# v2 "large" config — must match the pretrained run_config.json
ARCH = dict(in_channels=22, model_channels=64, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=3, time_emb_dim=768, dropout=0.1, attention_heads=8)

# Data directories: (path, label).  label 0 = normal, 1 = abnormal.
TRAIN_DIRS = [(f"{BASE}/train-normal", 0), (f"{BASE}/train-seizure", 1)]
TEST_DIRS = [(f"{BASE}/test-normal", 0), (f"{BASE}/test-seizure", 1)]

# Progressive unfreezing experiments.
#   n_down  = number of trailing entries of backbone.down_blocks to unfreeze
#   stem    = also unfreeze init_conv + time_mlp
# down_blocks has 7 entries for [1,2,4,8] x num_res_blocks=3 (4 res-block levels
# interleaved with 3 downsample convs). Run from least to most unfrozen.
UNFREEZE_EXPERIMENTS = [
    dict(name="head_only", n_down=0, stem=False),
    dict(name="unfreeze_1", n_down=1, stem=False),
    dict(name="unfreeze_2", n_down=2, stem=False),
    dict(name="unfreeze_3", n_down=3, stem=False),
    dict(name="unfreeze_5", n_down=5, stem=False),
    dict(name="unfreeze_all", n_down=99, stem=True),
]

SELECTION_METRIC = "f2"  # which validation metric picks the best epoch / experiment


# ================================================================
# DATA LOADING
# Training streams whole batch files SEQUENTIALLY (Lustre-friendly: ~hundreds of
# sequential reads instead of ~1M random per-sample mmap reads, which starved
# the GPU at ~6 h/epoch). Val/test use the simple per-sample memory-mapped path.
# ================================================================
def gather_files(dir_label_pairs):
    """Return [(path, label), ...] for all batch .npy files in the given dirs."""
    out = []
    for directory, label in dir_label_pairs:
        for f in sorted(_Path(directory).glob("*_batch_*.npy")):
            if "_labels" not in f.name:
                out.append((f, label))
    return out


class ClassInterleavedDataset(IterableDataset):
    """Sequential file reads, but classes interleaved at their natural ratio.

    The .npy files are single-class, so naively streaming them produces
    class-bursty batches. Instead we keep ONE stream per class, read whole files
    sequentially (Lustre-friendly) into per-class shuffle buffers, and draw each
    sample from a class chosen in proportion to its remaining samples — so every
    batch is uniformly class-mixed while I/O stays sequential.
    """

    def __init__(self, file_label_pairs, mean, std, buffer_size=4000, base_seed=42):
        self.mean = mean.reshape(-1, 1).astype(np.float32)
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
            if info is not None:
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

        def remaining(c):
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
    """Map-style per-sample memory-mapped dataset (used for val/test)."""

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
        self.labels = np.array([self.labels_per_file[fid] for fid, _ in self.index],
                               dtype=np.int64)
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


# ================================================================
# MODELS  (identical structure to finetune_binary_PW.py)
# ================================================================
class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = backbone
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

    def _encode_at_t(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        level_outputs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = module_list[0](h)
            else:
                for block in module_list:
                    if hasattr(block, 'forward') and 'time_emb' in block.forward.__code__.co_varnames:
                        h = block(h, t_emb)
                    else:
                        h = block(h)
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


# ================================================================
# UNFREEZING
# ================================================================
def apply_unfreezing(backbone, n_down, stem):
    """Freeze everything, then unfreeze the trailing `n_down` encoder entries
    (and optionally the stem). Returns the list of unfrozen backbone params."""
    for p in backbone.parameters():
        p.requires_grad = False

    unfrozen = []
    down = backbone.down_blocks
    if n_down > 0:
        for module_list in down[-n_down:]:
            for p in module_list.parameters():
                p.requires_grad = True
                unfrozen.append(p)
    if stem:
        for mod in (backbone.init_conv, backbone.time_mlp):
            for p in mod.parameters():
                p.requires_grad = True
                unfrozen.append(p)
    return unfrozen


# ================================================================
# METRICS
# ================================================================
def fbeta(precision, recall, beta=2.0, eps=1e-9):
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + eps)


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)          # sensitivity
    specificity = tn / (tn + fp + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    f1 = fbeta(precision, recall, beta=1.0)
    f2 = fbeta(precision, recall, beta=2.0)
    balanced_acc = 0.5 * (recall + specificity)

    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, probs)
        pc, rc, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rc, pc)
    else:
        roc_auc = pr_auc = 0.5

    return dict(threshold=float(threshold), accuracy=float(accuracy),
                precision=float(precision), recall=float(recall),
                specificity=float(specificity), f1=float(f1), f2=float(f2),
                balanced_acc=float(balanced_acc),
                roc_auc=float(roc_auc), pr_auc=float(pr_auc),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def find_best_threshold(labels, probs, metric="balanced_acc"):
    """Pick the operating threshold that maximizes the chosen metric.

    Defaults to balanced accuracy so the threshold doesn't collapse to a
    degenerate "predict everything positive" point (which is what maximizing
    F2/recall alone does).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    best_t, best_score, best_m = 0.5, -1.0, None
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(labels, probs, t)
        if m[metric] > best_score:
            best_t, best_score, best_m = t, m[metric], m
    return best_t, best_m


def batch_f1(preds, labels, eps=1e-6):
    p, l = preds.float(), labels.float()
    tp = (p * l).sum()
    fp = (p * (1 - l)).sum()
    fn = ((1 - p) * l).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)


# ================================================================
# TRAIN / EVAL
# ================================================================
def _autocast(device):
    # bf16 mixed precision on CUDA (no GradScaler needed for bf16); no-op on CPU
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_epoch(classifier, rl_layer, loader, optimizer, scheduler, loss_fn,
                device, trainable_params):
    classifier.train()
    rl_layer.train()
    classifier.backbone.eval()  # keep norm/dropout deterministic; unfrozen weights still get grads

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
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss.append(loss.item())
        rewards.append(reward.item())
        with torch.no_grad():
            preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    return float(np.mean(total_loss)), float(acc), float(np.mean(rewards))


@torch.no_grad()
def collect_probs(classifier, rl_layer, loader, loss_fn, device):
    classifier.eval()
    rl_layer.eval()
    total_loss, probs_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device):
            logits, features = classifier(x)
            adj, probs, _, _ = rl_layer(logits, features, training=False)
            total_loss += loss_fn(adj, y).item()
        probs_all.extend(probs.float().cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    return total_loss / max(len(loader), 1), np.array(probs_all), np.array(labels_all)


def fmt_metrics(m):
    return (f"F2:{m['f2']:.4f} F1:{m['f1']:.4f} Acc:{m['accuracy']:.4f} "
            f"Rec:{m['recall']:.4f} Prec:{m['precision']:.4f} Spec:{m['specificity']:.4f} "
            f"ROC:{m['roc_auc']:.4f} PR:{m['pr_auc']:.4f}")


# ================================================================
# ONE EXPERIMENT
# ================================================================
def run_experiment(exp, train_loader, val_loader, test_loader, class_weights,
                   steps_per_epoch, args, device, out_dir):
    name = exp["name"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}  (n_down={exp['n_down']}, stem={exp['stem']})")
    print("=" * 70)

    # Fresh backbone for every experiment (no carry-over between runs)
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
    try:
        ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
        backbone.load_state_dict(ckpt['model_state_dict'])
        print("  ✓ Backbone checkpoint loaded")
    except Exception as e:
        print(f"  ! Could not load checkpoint ({e}); using random backbone")
    backbone.eval()

    n_down = min(exp["n_down"], len(backbone.down_blocks))
    unfrozen_backbone = apply_unfreezing(backbone, n_down, exp["stem"])
    n_bb = sum(p.numel() for p in unfrozen_backbone)
    print(f"  Unfrozen backbone params: {n_bb:,}")

    classifier = EEGClassifier(backbone, feature_dim=256, dropout=0.4).to(device)
    agg_dim = backbone.model_channels * sum(backbone.channel_multipliers) * 5
    rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=0.1).to(device)

    # Optimizer: head + RL at full LR, unfrozen backbone at a lower LR
    param_groups = [
        {'params': list(classifier.classifier.parameters()), 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': list(rl_layer.parameters()), 'lr': args.lr, 'weight_decay': 1e-4},
    ]
    if unfrozen_backbone:
        param_groups.append({'params': unfrozen_backbone, 'lr': args.lr * args.backbone_lr_mult,
                             'weight_decay': 1e-4})
    optimizer = optim.AdamW(param_groups)
    max_lrs = [args.lr, args.lr] + ([args.lr * args.backbone_lr_mult] if unfrozen_backbone else [])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='cos')

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable_params = [p for g in param_groups for p in g['params']]

    best_val_score = -1.0
    best_state = None
    patience_counter = 0
    history = []

    # Create the output dir up front so we can checkpoint the best model to disk on
    # every improvement (crash/timeout-safe). Previously best_state lived only in RAM
    # until the experiment finished, so a mid-experiment failure lost it entirely.
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)  # reshuffle streaming order
        tr_loss, tr_acc, tr_reward = train_epoch(
            classifier, rl_layer, train_loader, optimizer, scheduler, loss_fn,
            device, trainable_params)
        val_loss, val_probs, val_labels = collect_probs(
            classifier, rl_layer, val_loader, loss_fn, device)
        val_m = compute_metrics(val_labels, val_probs, threshold=0.5)
        score = val_m[SELECTION_METRIC]
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=val_loss, **{f"val_{k}": v for k, v in val_m.items()}))

        print(f"  Epoch {epoch:3d} | tr_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val {fmt_metrics(val_m)}")

        if score > best_val_score:
            best_val_score = score
            best_state = {
                'classifier': {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
                'rl_layer': {k: v.cpu().clone() for k, v in rl_layer.state_dict().items()},
            }
            patience_counter = 0
            # Persist the best model immediately (survives a later crash/timeout).
            # The final save below re-writes this with the tuned threshold added.
            torch.save({'classifier': best_state['classifier'],
                        'rl_layer': best_state['rl_layer'],
                        'arch': ARCH, 'exp': exp, 'epoch': epoch,
                        'best_val_score': float(best_val_score),
                        f'best_val_{SELECTION_METRIC}': float(best_val_score)},
                       exp_dir / "best.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best and evaluate on val (to pick threshold) + test
    classifier.load_state_dict({k: v.to(device) for k, v in best_state['classifier'].items()})
    rl_layer.load_state_dict({k: v.to(device) for k, v in best_state['rl_layer'].items()})

    _, val_probs, val_labels = collect_probs(classifier, rl_layer, val_loader, loss_fn, device)
    best_t, _ = find_best_threshold(val_labels, val_probs, metric="balanced_acc")

    _, test_probs, test_labels = collect_probs(classifier, rl_layer, test_loader, loss_fn, device)
    test_default = compute_metrics(test_labels, test_probs, threshold=0.5)
    test_tuned = compute_metrics(test_labels, test_probs, threshold=best_t)

    print(f"\n  >>> {name} TEST @0.5  : {fmt_metrics(test_default)}")
    print(f"  >>> {name} TEST @{best_t:.2f} : {fmt_metrics(test_tuned)}")

    # Persist final best model (exp_dir already created above). This re-writes the
    # per-improvement checkpoint, now including the tuned decision threshold.
    torch.save({'classifier': best_state['classifier'], 'rl_layer': best_state['rl_layer'],
                'arch': ARCH, 'exp': exp, 'best_threshold': best_t},
               exp_dir / "best.pth")
    with open(exp_dir / "results.json", 'w') as f:
        json.dump(dict(experiment=exp, best_val_score=float(best_val_score),
                       best_threshold=float(best_t),
                       test_default=test_default, test_tuned=test_tuned,
                       history=history), f, indent=2)

    return dict(name=name, best_val_score=float(best_val_score), best_threshold=float(best_t),
                test_default=test_default, test_tuned=test_tuned,
                unfrozen_backbone_params=int(n_bb))


# ================================================================
# MAIN
# ================================================================
def build_loaders(args, device):
    mean = np.load(_Path(NORM_STATS_DIR) / "mean.npy").astype(np.float32)
    std = np.load(_Path(NORM_STATS_DIR) / "std.npy").astype(np.float32)

    if args.dryrun:
        n = 32
        Xd = torch.randn(2 * n, ARCH['in_channels'], 1280)
        yd = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
        train_ds = TensorDataset(Xd, yd); val_ds = TensorDataset(Xd[:16], yd[:16])
        test_ds = TensorDataset(Xd[:16], yd[:16])
        w0 = w1 = 1.0
        steps_per_epoch = max(1, (2 * n) // args.batch_size)
    else:
        print("=" * 70 + "\nINDEXING DATA (sequential block-shuffle for train)\n" + "=" * 70)
        all_files = gather_files(TRAIN_DIRS)
        # File-level stratified validation split (keeps val files out of train)
        split_rng = random.Random(42)
        by_class = {0: [], 1: []}
        for f, l in all_files:
            by_class[l].append((f, l))
        train_files, val_files = [], []
        for l, items in by_class.items():
            split_rng.shuffle(items)
            k = max(1, int(round(args.val_frac * len(items))))
            val_files += items[:k]
            train_files += items[k:]

        # Count samples per file (header reads) for class weights + LR schedule
        n0 = n1 = 0
        for f, l in tqdm(train_files, desc="  Counting train samples"):
            nn_ = np.load(f, mmap_mode='r').shape[0]
            if l == 0:
                n0 += nn_
            else:
                n1 += nn_
        n_train = n0 + n1
        steps_per_epoch = max(1, n_train // args.batch_size)

        train_ds = ClassInterleavedDataset(train_files, mean, std, buffer_size=args.shuffle_buffer)
        val_ds = LazyEEGDataset(val_files, mean, std)
        test_ds = LazyEEGDataset(gather_files(TEST_DIRS), mean, std)

        w0 = n_train / (2.0 * max(n0, 1))
        w1 = n_train / (2.0 * max(n1, 1))
        print(f"\n✓ Train files {len(train_files)} | {n_train:,} samples "
              f"(normal {n0:,} / abnormal {n1:,})")
        print(f"✓ Val   files {len(val_files)} | {len(val_ds):,} samples")
        print(f"✓ Test  | {len(test_ds):,} samples")
        print(f"✓ Steps/epoch {steps_per_epoch:,} | class weights "
              f"normal={w0:.3f} abnormal={w1:.3f}\n")

    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    eval_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        eval_kw["persistent_workers"] = True
        eval_kw["prefetch_factor"] = 4
    # Train is an IterableDataset (shuffles internally); persistent_workers=False
    # so set_epoch() reshuffles each epoch.
    train_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        train_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, **train_kw)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, **eval_kw)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False, **eval_kw)
    return train_loader, val_loader, test_loader, class_weights, steps_per_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="LR multiplier for unfrozen backbone params (relative to --lr)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--shuffle-buffer", type=int, default=4000,
                   help="per-worker sample shuffle buffer for the training stream")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--experiments", type=str, default="all",
                   help="comma-separated experiment names, or 'all'")
    p.add_argument("--dryrun", action="store_true")
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.experiments == "all":
        experiments = UNFREEZE_EXPERIMENTS
    else:
        wanted = {s.strip() for s in args.experiments.split(",")}
        experiments = [e for e in UNFREEZE_EXPERIMENTS if e["name"] in wanted]
    if not experiments:
        raise SystemExit(f"No matching experiments for '{args.experiments}'")

    train_loader, val_loader, test_loader, class_weights, steps_per_epoch = build_loaders(args, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _Path(OUT_ROOT) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs -> {out_dir}/")

    results = []
    for exp in experiments:
        res = run_experiment(exp, train_loader, val_loader, test_loader,
                             class_weights, steps_per_epoch, args, device, out_dir)
        results.append(res)

    # Summary
    print("\n" + "=" * 70 + "\nSUMMARY (test @ tuned threshold)\n" + "=" * 70)
    header = f"{'experiment':<14}{'F2':>8}{'F1':>8}{'Acc':>8}{'Recall':>8}{'ROC':>8}{'PR':>8}"
    print(header)
    for r in results:
        m = r["test_tuned"]
        print(f"{r['name']:<14}{m['f2']:>8.4f}{m['f1']:>8.4f}{m['accuracy']:>8.4f}"
              f"{m['recall']:>8.4f}{m['roc_auc']:>8.4f}{m['pr_auc']:>8.4f}")

    best = max(results, key=lambda r: r["test_tuned"][SELECTION_METRIC])
    print(f"\n*** BEST: {best['name']} | test {fmt_metrics(best['test_tuned'])} "
          f"(threshold {best['best_threshold']:.2f}) ***")

    with open(out_dir / "summary.json", 'w') as f:
        json.dump(dict(selection_metric=SELECTION_METRIC, best=best['name'],
                       results=results, args=vars(args)), f, indent=2)
    print(f"\n✓ All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
