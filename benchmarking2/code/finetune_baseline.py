#!/usr/bin/env python3
"""Fine-tune BIOT or LaBraM on the benchmark's exact TUAB subset.

Each model is adapted with ITS OWN authors' published recipe (optimiser, learning rate,
weight decay, batch size, schedule), but on identical data, identical splits and with
identical metrics, so the comparison isolates the model rather than the pipeline. We do
not reuse their training scripts because those bring their own splits and preprocessing,
which is precisely the confound this benchmark exists to remove.

Published recipes followed here:
  BIOT   -- run_binary_supervised.py defaults: Adam, lr 1e-3, wd 1e-5, batch 512,
            BCE on a single logit, BIOTClassifier = pretrained encoder + linear head.
  LaBraM -- README TUAB command: AdamW, lr 5e-4, wd 0.05, batch 64, layer decay 0.65,
            drop_path 0.1, 5 warmup epochs, cosine schedule, and their own
            LayerDecayValueAssigner so the layer-wise LR decay is theirs, not ours.

Selection is on validation AUROC for every model, matching probe.py, mlp_head.py and
finetune_diffeeg.py.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

B = Path("/home/abdulh/scratch/benchmarking2")
sys.path.insert(0, str(B / "code"))
import data_adapter as DA
from extract_features import get_splits

N_BOOT, SEED = 1000, 0
CFG = {
    "biot":   dict(lr=1e-3, wd=1e-5, batch=256, epochs=50, patience=8, warmup=0,
                   layer_decay=None, opt="adam"),
    "labram": dict(lr=5e-4, wd=0.05, batch=64,  epochs=50, patience=8, warmup=5,
                   layer_decay=0.65, opt="adamw"),
}


class ViewDataset(Dataset):
    """Lazy per-sample loading + the model's published input view.

    Every view transform in data_adapter is per-sample (bipolar derivation, resampling,
    per-sample percentile normalisation, constant scaling), so applying them one sample
    at a time is numerically identical to the batched path used for feature extraction.
    """
    def __init__(self, triples, view, ds="tuab5s"):
        self.t, self.view, self.cfg = list(triples), view, DA.DATASETS[ds]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        f, r, lab = self.t[i]
        s = np.asarray(np.load(f, mmap_mode="r")[r], dtype=np.float32)
        if s.ndim == 2 and s.shape[0] != 22 and s.shape[1] == 22:
            s = s.T
        x = s[None]                                     # (1, 22, 1280)
        if self.view == "biot":
            x = DA.to_200hz(DA.to_bipolar(x))[:, DA.BIOT16_IDX, :]
            x = x / (np.quantile(np.abs(x), 0.95, axis=-1, keepdims=True) + 1e-8)
        elif self.view == "labram":
            x = DA.to_200hz(x[:, :DA.LABRAM_N_CH, :]) / 100.0
        else:
            raise ValueError(self.view)
        return torch.from_numpy(x[0]).float(), torch.tensor(float(lab))


def metrics(y, p, prob):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, p)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)),
            "f1_pos": float(f1_score(y, p, zero_division=0)),
            "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
            "precision": float(precision_score(y, p, zero_division=0)),
            "sensitivity": float(recall_score(y, p, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1))}


def build_biot(dev):
    sys.path.insert(0, str(B / "BIOT"))
    from model.biot import BIOTClassifier
    m = BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=1,
                       n_channels=18, n_fft=200, hop_length=100)
    sd = torch.load(B / "BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt",
                    map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("biot."):] if k.startswith("biot.") else k): v for k, v in sd.items()}
    miss, unexp = m.biot.load_state_dict(sd, strict=False)
    print(f"  BIOT encoder loaded (missing {len(miss)}, unexpected {len(unexp)}); "
          f"total {sum(p.numel() for p in m.parameters()):,} params", flush=True)
    return m.to(dev), None


def build_labram(dev):
    sys.path.insert(0, str(B / "LaBraM"))
    import modeling_finetune  # noqa: F401
    from timm.models import create_model
    import utils as labram_utils
    m = create_model("labram_base_patch200_200", pretrained=False, num_classes=1,
                     drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=False, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=False)
    ck = torch.load(B / "LaBraM/checkpoints/labram-base.pth", map_location="cpu",
                    weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")} or sd
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("head.", "lm_head."))
          and "relative_position_index" not in k
          and k not in ("mask_token", "logit_scale")}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  LaBraM loaded (missing {len(miss)}, unexpected {len(unexp)}); "
          f"total {sum(p.numel() for p in m.parameters()):,} params", flush=True)
    ch = torch.tensor(labram_utils.get_input_chans(DA.LABRAM_CH_NAMES), dtype=torch.long)
    return m.to(dev), ch.to(dev)


def make_optimizer(model, name, cfg):
    if name == "biot":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["wd"]), None
    # LaBraM: their own layer-wise LR decay
    sys.path.insert(0, str(B / "LaBraM"))
    from optim_factory import create_optimizer, LayerDecayValueAssigner
    n_layers = model.get_num_layers()
    assigner = LayerDecayValueAssigner(
        list(cfg["layer_decay"] ** (n_layers + 1 - i) for i in range(n_layers + 2)))
    args = SimpleNamespace(opt="adamw", lr=cfg["lr"], weight_decay=cfg["wd"],
                           opt_eps=1e-8, opt_betas=None, momentum=0.9,
                           weight_decay_end=None)
    opt = create_optimizer(args, model, get_num_layer=assigner.get_layer_id,
                           get_layer_scale=assigner.get_scale)
    return opt, assigner


def forward(model, name, x, ch):
    if name == "biot":
        return model(x).squeeze(-1)
    b, c, L = x.shape
    return model(x.reshape(b, c, L // DA.LABRAM_PATCH, DA.LABRAM_PATCH),
                 input_chans=ch).squeeze(-1)


@torch.no_grad()
def predict(model, name, dl, ch, dev):
    model.eval(); P, Y = [], []
    for x, y in dl:
        p = torch.sigmoid(forward(model, name, x.to(dev, non_blocking=True), ch))
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["biot", "labram"])
    a = ap.parse_args()
    cfg = CFG[a.model]
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = B / "results" / f"{a.model}_ft"; out.mkdir(parents=True, exist_ok=True)
    ckpt_p, best_p = out / "ckpt.pth", out / "best.pth"

    sp = get_splits("tuab5s", DA)
    print({k: len(v) for k, v in sp.items()}, flush=True)
    dls = {k: DataLoader(ViewDataset(sp[k], a.model), batch_size=cfg["batch"],
                         shuffle=(k == "train"), num_workers=6, pin_memory=True,
                         persistent_workers=True, drop_last=(k == "train"))
           for k in ("train", "val", "test")}

    model, ch = (build_biot(dev) if a.model == "biot" else build_labram(dev))
    opt, _ = make_optimizer(model, a.model, cfg)
    ytr = np.array([l for _, _, l in sp["train"]])
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                      dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    steps = len(dls["train"])
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(ep, it):
        t = ep + it / steps
        if t < cfg["warmup"]:
            f = t / max(cfg["warmup"], 1e-8)
        else:
            f = 0.5 * (1 + math.cos(math.pi * (t - cfg["warmup"]) /
                                    max(cfg["epochs"] - cfg["warmup"], 1e-8)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    start, best, bad, hist = 0, -1.0, 0, []
    if ckpt_p.exists():
        c = torch.load(ckpt_p, map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        start, best, bad, hist = c["epoch"] + 1, c["best"], c["bad"], c.get("hist", [])
        print(f"resumed at epoch {start} (best val AUROC {best:.4f})", flush=True); del c

    for ep in range(start, cfg["epochs"]):
        model.train(); t0, losses = time.time(), []
        for it, (x, y) in enumerate(dls["train"]):
            set_lr(ep, it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(forward(model, a.model, x, ch), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); continue
            opt.step(); losses.append(loss.item())
        pv, yv = predict(model, a.model, dls["val"], ch, dev)
        vauc = roc_auc_score(yv, pv)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_auc": float(vauc)})
        print(f"  ep {ep:02d} loss {np.mean(losses):.4f}  val AUROC {vauc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best, bad = vauc, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": float(vauc)}, best_p)
            print(f"  >> new best (epoch {ep})", flush=True)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  early stop at epoch {ep}", flush=True); break
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "hist": hist}, ckpt_p)

    bs = torch.load(best_p, map_location=dev); model.load_state_dict(bs["model"])
    prob, yte = predict(model, a.model, dls["test"], ch, dev)
    pred = (prob >= 0.5).astype(int)
    point = metrics(yte, pred, prob)
    rng = np.random.default_rng(0); boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[i])) < 2: continue
        boot.append(metrics(yte[i], pred[i], prob[i]))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)),
              float(np.percentile([b[k] for b in boot], 97.5))) for k in point}
    res = {"model": a.model, "protocol": "fine-tuned (authors' published recipe)",
           "selection": "validation AUROC", "config": cfg,
           "best_epoch": int(bs["epoch"]), "val_auc": float(bs["val_auc"]),
           "n": {k: len(v) for k, v in sp.items()},
           "test_prevalence": float(yte.mean()), "metrics": point, "ci95": ci,
           "confusion": confusion_matrix(yte, pred, labels=[0, 1]).tolist(), "history": hist}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    ks = ["accuracy", "roc_auc", "pr_auc", "f1_weighted", "f1_macro", "sensitivity", "specificity"]
    print("\n" + "".join(f"{k:>13}" for k in ks))
    print("".join(f"{point[k]:>13.4f}" for k in ks))
    print(f"\nsaved -> {out/'result.json'}")


if __name__ == "__main__":
    main()
