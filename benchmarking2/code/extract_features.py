#!/usr/bin/env python3
"""Extract frozen representations from DiffEEG or EEGDM on our datasets.

    python extract_features.py --model eegdm   --dataset tuab5s
    python extract_features.py --model diffeeg --dataset tusz

Both backbones are frozen and used exactly as published; only feature extraction
happens here.

Pooling
-------
EEGDM's reducer emits (B, 20 layers, 1, 5 windows, 1, 22 channels, 128 feats) =
281,600 values per sample -- far too large to store for ~10^5 samples, and not what
their Latent Fusion Transformer consumes as a flat vector anyway. We mean-pool over
the window and EEG-channel axes, leaving 20 x 128 = 2,560 dims.

That mirrors what DiffEEG's own downstream head does: global-average-pool each
encoder level over time, giving 5 timesteps x 480 = 2,400 dims. So both models end up
with a time-pooled, channel-aggregated descriptor of near-identical size, which keeps
the linear probe from simply rewarding whichever model emits more numbers.

Sampling
--------
The corpora are large (TUSZ: 1.34M train). We take a fixed class-balanced training
subsample and a class-PROPORTIONAL test subsample -- proportional so the test set keeps
its real prevalence (6.7% seizure for TUSZ), which is the clinically meaningful setting.
Indices are drawn once with a fixed seed and reused for both models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

B = Path("/home/abdulh/scratch/benchmarking2")
OUT = B / "results" / "features"
TRAIN_CAP, VAL_CAP, TEST_CAP = 100_000, 20_000, 50_000

# Per-dataset sampling policy. TUAB is huge and near-balanced, so a class-balanced
# subsample is both affordable and representative. The external corpora are different:
# Siena carries only 519 seizure segments in 82,791 training samples (0.63%) and Bonn
# only 400 training segments in total, so a class-balanced cap would discard ~99% of
# Siena and over half of Bonn. For those we keep every segment and let the classifier's
# balanced class weighting handle the skew, which preserves all positives.
SPLIT_POLICY = {
    "tuab5s": dict(train=(TRAIN_CAP, True),  val=(VAL_CAP, True),  test=(TEST_CAP, False)),
    "tusz":   dict(train=(TRAIN_CAP, True),  val=(VAL_CAP, True),  test=(TEST_CAP, False)),
    "bonn":   dict(train=(None, False),      val=(None, False),    test=(None, False)),
    "siena":  dict(train=(None, False),      val=(None, False),    test=(None, False)),
}
SEED = 0
# EEGDM's S4 layers use FFT convolutions over (20 layers x 22 ch x 1000 samples),
# which OOMs an A100 at batch 128; DiffEEG's 1-D U-Net is far lighter.
BATCH = {"eegdm": 32, "diffeeg": 128, "biot": 256, "labram": 128, "cbramod": 128,
         "eegmamba": 64,
         "diffeeg_noised": 128, "diffeeg_win": 128, "diffeeg_tok": 128}   # eegdm: max with pykeops


def choose(index, cap, balanced, n_class, seed):
    labs = np.array([l for _, _, l in index])
    rng = np.random.default_rng(seed)
    if cap >= len(index):
        return list(range(len(index)))
    if balanced:
        per = cap // n_class
        keep = np.concatenate([rng.choice(np.where(labs == c)[0],
                                          size=min(per, int((labs == c).sum())),
                                          replace=False) for c in range(n_class)])
    else:  # proportional -- preserves true prevalence
        keep = rng.choice(len(index), size=cap, replace=False)
    return sorted(keep.tolist())


def get_splits(ds, adapter):
    n_class = len(adapter.DATASETS[ds]["classes"])
    tr_idx = adapter.build_index(ds, "train")
    te_idx = adapter.build_index(ds, "test")
    rng = np.random.default_rng(SEED)
    # Stratified validation split, so a rare-positive corpus keeps positives on both
    # sides. A plain 10% permutation can leave the validation fold with almost none.
    labs = np.array([l for _, _, l in tr_idx])
    va_sel = np.zeros(len(tr_idx), dtype=bool)
    for c in range(n_class):
        ci = np.where(labs == c)[0]
        k = max(1, int(round(0.1 * len(ci))))
        va_sel[rng.choice(ci, size=min(k, len(ci)), replace=False)] = True
    va_pool = [tr_idx[i] for i in np.where(va_sel)[0]]
    tr_pool = [tr_idx[i] for i in np.where(~va_sel)[0]]

    pol = SPLIT_POLICY[ds]
    def take(pool, key, seed):
        cap, bal = pol[key]
        if cap is None:
            return list(pool)
        return [pool[i] for i in choose(pool, cap, bal, n_class, seed)]
    return {"train": take(tr_pool, "train", SEED),
            "val":   take(va_pool, "val",   SEED + 1),
            "test":  take(te_idx,  "test",  SEED + 2)}


@torch.no_grad()
def eegdm_extractor(dev):
    import sys
    sys.path.insert(0, str(B / "EEGDM_ref"))
    _o = torch.load
    torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
    from model.classifier import LatentActivityExtractor, LatentActivityReducer
    from model.diffusion_model_pl import PLDiffusionModel
    from src.util import staged_mu_law

    R = dict(query=["gate"], reduce=["std"], rescale=False, L=1000,
             window_size=200, window_step=200, pool_merge="share", multi_query_merge="seq")
    dm = PLDiffusionModel.load_from_checkpoint(
        B / "EEGDM_ref/checkpoint/pretrain/backbone.ckpt", map_location=dev)
    net = torch.nn.Sequential(
        LatentActivityExtractor(model=dm.ema.ema_model, diffusion_t=1,
                                query=R["query"], use_cond=None),
        LatentActivityReducer(**R)).to(dev).eval()
    print(f"  EEGDM backbone params: {sum(p.numel() for p in dm.parameters()):,}", flush=True)

    @torch.no_grad()
    def run(xb):
        xb = np.stack([staged_mu_law(x.copy()) for x in xb])   # their loader-side transform
        t = torch.from_numpy(xb).float().to(dev)
        o = net((t, None))                       # (B, 20, 1, 5, 1, 22, 128)
        o = o.squeeze(4).squeeze(2)              # (B, 20, 5, 22, 128)
        return o.mean(dim=(2, 3)).flatten(1)     # pool windows + channels -> (B, 20*128)
    return run


@torch.no_grad()
def diffeeg_extractor(dev):
    import sys
    sys.path.insert(0, "/home/abdulh/scratch")
    from Diff_EEG_train import DeepEnhancedEEGDiffusionModel
    ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)
    bb = DeepEnhancedEEGDiffusionModel(**ARCH).to(dev)
    ck = torch.load("/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth",
                    map_location=dev)
    bb.load_state_dict(ck["model_state_dict"]); del ck
    bb.eval()
    pool = torch.nn.AdaptiveAvgPool1d(1).to(dev)
    probes = torch.tensor([50, 250, 500, 750, 950], dtype=torch.long, device=dev)
    print(f"  DiffEEG backbone params: {sum(p.numel() for p in bb.parameters()):,}", flush=True)

    def enc(x, t):
        te = bb.time_mlp(t)
        h = bb.init_conv(x)
        outs = []
        for ml in bb.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], torch.nn.Conv1d):
                h = ml[0](h)
            else:
                for blk in ml:
                    h = (blk(h, te) if "time_emb" in blk.forward.__code__.co_varnames else blk(h))
                outs.append(pool(h).squeeze(-1))
        return torch.cat(outs, 1)

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        return torch.cat([enc(t, ts.expand(t.shape[0])) for ts in probes], 1)
    return run



def diffeeg_win_extractor(dev):
    """Same frozen V1 backbone as diffeeg_extractor, windowed pooling instead of GAP.

    This isolates the READOUT. Backbone, checkpoint and input are identical to the
    published configuration; the only change is what we do with the encoder
    activations. Two differences from diffeeg_extractor:

      * one timestep (t=50) instead of five. Our ablation measured the five views to
        correlate >0.995 and span 0.0011 AUROC, so they are near-duplicates; EEGDM
        likewise extracts at a single small t.
      * per encoder level, the time axis is split into 5 equal windows and we keep the
        mean AND the std of each, instead of averaging the whole segment away. Global
        averaging cannot distinguish a rhythmic burst from flat background, which is
        precisely the distinction seizure detection rests on. EEGDM std-pools its gate
        activations over 200-sample windows for the same reason.

    Dims: (32+64+128+256) channels x 5 windows x {mean,std} = 4,800.
    """
    import sys
    sys.path.insert(0, "/home/abdulh/scratch")
    from Diff_EEG_train import DeepEnhancedEEGDiffusionModel
    ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)
    bb = DeepEnhancedEEGDiffusionModel(**ARCH).to(dev)
    ck = torch.load("/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth",
                    map_location=dev)
    bb.load_state_dict(ck["model_state_dict"]); del ck
    bb.eval()
    W = 5
    probe_t = torch.tensor(50, dtype=torch.long, device=dev)
    print(f"  DiffEEG backbone params: {sum(p.numel() for p in bb.parameters()):,}"
          f" | windowed readout: {W} windows, mean+std, t=50", flush=True)

    def win_pool(h):
        b, c, L = h.shape
        assert L % W == 0, f"level length {L} not divisible by {W} windows"
        hw = h.view(b, c, W, L // W)
        return torch.cat([hw.mean(-1), hw.std(-1)], dim=2).reshape(b, -1)

    def enc(x, t):
        te = bb.time_mlp(t)
        h = bb.init_conv(x)
        outs = []
        for ml in bb.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], torch.nn.Conv1d):
                h = ml[0](h)
            else:
                for blk in ml:
                    h = (blk(h, te) if "time_emb" in blk.forward.__code__.co_varnames else blk(h))
                outs.append(win_pool(h))
        return torch.cat(outs, 1)

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        return enc(t, probe_t.expand(t.shape[0]))
    return run



def diffeeg_tok_extractor(dev):
    """V1 backbone, windowed features kept as a TOKEN SEQUENCE for a fusion head.

    diffeeg_win_extractor flattens the windowed statistics into one vector for a
    linear probe. This variant keeps the window axis, so a fusion transformer can
    attend over time windows and encoder levels the way EEGDM's Latent Fusion
    Transformer attends over its windows, channels and blocks.

    Output per sample: [W, 960] with W=8 windows. Each window holds, concatenated
    by encoder level, [mean(C_l), std(C_l)] with C_l in {32,64,128,256}, so the
    960 axis splits at offsets 0:64 | 64:192 | 192:448 | 448:960 into the four
    levels. A head can therefore treat either the window or the (window, level)
    pair as a token.

    NOTE on what this can and cannot test: V1's init_conv mixes the 22 electrodes
    into feature maps immediately, so no channel axis survives to tokenise. This
    isolates the TIME-WINDOW half of the V3 readout only; the cross-channel half
    needs the V3 backbone.
    """
    import sys
    sys.path.insert(0, "/home/abdulh/scratch")
    from Diff_EEG_train import DeepEnhancedEEGDiffusionModel
    ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)
    bb = DeepEnhancedEEGDiffusionModel(**ARCH).to(dev)
    ck = torch.load("/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth",
                    map_location=dev)
    bb.load_state_dict(ck["model_state_dict"]); del ck
    bb.eval()
    W = 8
    probe_t = torch.tensor(50, dtype=torch.long, device=dev)
    print(f"  DiffEEG backbone params: {sum(p.numel() for p in bb.parameters()):,}"
          f" | token readout: {W} windows x 960 (mean+std, 4 levels), t=50", flush=True)

    def win_stats(h):
        b, c, L = h.shape
        assert L % W == 0, f"level length {L} not divisible by {W} windows"
        hw = h.view(b, c, W, L // W)
        return torch.cat([hw.mean(-1), hw.std(-1)], dim=1).permute(0, 2, 1)   # [B, W, 2C]

    def enc(x, t):
        te = bb.time_mlp(t)
        h = bb.init_conv(x)
        outs = []
        for ml in bb.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], torch.nn.Conv1d):
                h = ml[0](h)
            else:
                for blk in ml:
                    h = (blk(h, te) if "time_emb" in blk.forward.__code__.co_varnames else blk(h))
                outs.append(win_stats(h))
        return torch.cat(outs, dim=2)      # [B, W, 960]

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        return enc(t, probe_t.expand(t.shape[0]))
    return run


@torch.no_grad()
def biot_extractor(dev):
    """Frozen BIOT encoder -> 256-d embedding.

    Uses their released EEG-six-datasets-18-channels checkpoint via BIOTEncoder, whose
    forward() already returns a pooled [batch, emb_size] vector, so no extra pooling is
    imposed by us. n_channels=18 matches the checkpoint's channel-token table; we feed
    the 16 TCP channels their README lists for that montage, which the encoder handles
    because channel tokens are looked up per channel.
    """
    import sys
    sys.path.insert(0, str(B / "BIOT"))
    from model.biot import BIOTEncoder

    ck = B / "BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt"
    enc = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18,
                      n_fft=200, hop_length=100)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = { (k[len("biot."):] if k.startswith("biot.") else k): v for k, v in sd.items() }
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"  BIOT loaded: {sum(p.numel() for p in enc.parameters()):,} params "
          f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    enc = enc.to(dev).eval()

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        return enc(t)
    return run


@torch.no_grad()
def labram_extractor(dev):
    """Frozen LaBraM base encoder -> 200-d embedding.

    forward_features(..., input_chans=...) already mean-pools the patch tokens, so the
    pooling is theirs, not ours. input_chans indexes their standard_1020 table (offset
    by one for the cls token), which is how LaBraM applies channel-position embeddings.
    """
    import sys
    sys.path.insert(0, str(B / "LaBraM"))
    import modeling_finetune  # noqa: F401  (registers labram_base_patch200_200)
    from timm.models import create_model
    import utils as labram_utils
    import data_adapter as A

    # Config read from the checkpoint's OWN stored args, not from the finetuning
    # script's defaults (which differ): rel_pos_bias=False, abs_pos_emb=True,
    # layer_scale_init_value=0.1. qkv_bias=False because the weights carry a fused
    # attn.qkv.weight with no separate q_bias/v_bias. abs_pos_emb must be on, since
    # forward_features indexes self.pos_embed whenever input_chans is supplied.
    model = create_model("labram_base_patch200_200", pretrained=False, num_classes=0,
                         drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
                         drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                         use_rel_pos_bias=False, use_abs_pos_emb=True,
                         init_values=0.1, qkv_bias=False)
    # Load exactly as run_class_finetuning.py does: keep only the `student.` weights,
    # drop the head and any relative_position_index, then load non-strict. Their script
    # does NOT map the pretrain `norm` onto the finetune `fc_norm`, so fc_norm stays at
    # LayerNorm's default (weight=1, bias=0) -- the same state their own fine-tuning
    # starts from. We do not deviate from that.
    # weights_only=False: their checkpoint pickles numpy scalars inside `args`,
    # which torch>=2.6 rejects under the new default. It is the authors' own file.
    ck = torch.load(B / "LaBraM/checkpoints/labram-base.pth", map_location="cpu",
                    weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")} or sd
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("head.", "lm_head."))
          and "relative_position_index" not in k
          and k not in ("mask_token", "logit_scale")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  LaBraM loaded: {sum(p.numel() for p in model.parameters()):,} params "
          f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    model = model.to(dev).eval()

    ch = torch.tensor(labram_utils.get_input_chans(A.LABRAM_CH_NAMES),
                      dtype=torch.long, device=dev)

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        b, c, L = t.shape
        t = t.reshape(b, c, L // A.LABRAM_PATCH, A.LABRAM_PATCH)   # their documented layout
        return model.forward_features(t, input_chans=ch)
    return run



@torch.no_grad()
def cbramod_extractor(dev):
    """Frozen CBraMod encoder -> 200-d embedding.

    Input is the 16-channel TCP bipolar montage at 200 Hz reshaped into one-second
    patches, which is the layout their TUAB configuration uses (models/model_for_tuab.py
    builds the backbone with in_dim=out_dim=d_model=200). Pooling is theirs, not ours:
    their `avgpooling_patch_reps` classifier applies AdaptiveAvgPool2d((1,1)) over the
    channel and patch axes, which is a mean over both, leaving 200 dims. We set
    proj_out = Identity exactly as their quick_example.py does before reading features.
    """
    import sys
    sys.path.insert(0, str(B / "CBraMod"))
    from models.cbramod import CBraMod
    import data_adapter as A

    net = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                  seq_len=30, n_layer=12, nhead=8)
    sd = torch.load(B / "CBraMod/pretrained_weights/pretrained_weights.pth",
                    map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  CBraMod loaded: {sum(p.numel() for p in net.parameters()):,} params "
          f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    net.proj_out = torch.nn.Identity()
    net = net.to(dev).eval()

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        b, c, L = t.shape
        t = t.reshape(b, c, L // A.LABRAM_PATCH, A.LABRAM_PATCH)   # 200-sample patches
        o = net(t)                       # (B, 16, patches, 200)
        return o.mean(dim=(1, 2))        # their avgpooling_patch_reps -> (B, 200)
    return run



def eegmamba_extractor(dev):
    """Frozen EEGMamba encoder -> 200-d embedding.

    EEGMamba (Neural Networks 2025, wjq-learning) is implemented on top of the
    CBraMod codebase and inherits its input contract exactly: the 16-channel TCP
    bipolar montage at 200 Hz, reshaped into one-second patches and divided by 100
    (see their datasets/tuab_dataset.py, which does `data.reshape(16, 10, 200)` then
    `data/100`). So it reuses our CBraMod view unchanged rather than a new data path.

    Pooling is theirs: `avgpooling_patch_reps` in models/model_for_tuab.py applies
    AdaptiveAvgPool2d((1,1)) over the channel and patch axes, a mean over both,
    leaving 200 dims -- the same descriptor width as CBraMod, so the two are
    directly comparable.

    Model construction lives in EEGMamba/eegmamba_loader.py, shared with the smoke
    test, and raises on any checkpoint key mismatch.
    """
    import sys
    sys.path.insert(0, str(B / "EEGMamba"))
    from eegmamba_loader import load_eegmamba
    import data_adapter as A

    net = load_eegmamba(dev)

    @torch.no_grad()
    def run(xb):
        t = torch.from_numpy(xb).float().to(dev)
        b, c, L = t.shape
        t = t.reshape(b, c, L // A.LABRAM_PATCH, A.LABRAM_PATCH)   # 200-sample patches
        o = net(t)                       # (B, 16, patches, 200)
        return o.mean(dim=(1, 2))        # their avgpooling_patch_reps -> (B, 200)
    return run


@torch.no_grad()
def diffeeg_noised_extractor(dev):
    """DiffEEG features WITH the input actually corrupted to each probe timestep.

    The published configuration (diffeeg_extractor) feeds the clean segment and lets the
    timestep act only through FiLM conditioning. Measured on TUSZ, the five resulting
    blocks correlate at 0.995-1.000 and a single block matches all five to within 0.002
    AUROC, so the multi-timestep concatenation is nearly redundant.

    This variant restores the condition the design assumes: before each pass the segment
    is noised to level t using the same cosine schedule the backbone was pre-trained with
    (train_enhanced_diffusion uses beta_schedule='cosine'), so a timestep index and the
    corruption actually present in the input agree, as they did during pre-training.
    Noise is drawn from a fixed generator so the descriptor stays reproducible.
    """
    import sys, math
    sys.path.insert(0, "/home/abdulh/scratch")
    from Diff_EEG_train import DeepEnhancedEEGDiffusionModel
    ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)
    bb = DeepEnhancedEEGDiffusionModel(**ARCH).to(dev)
    ck = torch.load("/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth",
                    map_location=dev)
    bb.load_state_dict(ck["model_state_dict"]); del ck
    bb.eval()

    # cosine schedule, identical to ImprovedDiffusionScheduler(beta_schedule='cosine')
    Tt, sN = 1000, 0.008
    x = torch.linspace(0, Tt, Tt + 1, device=dev)
    ac = torch.cos(((x / Tt) + sN) / (1 + sN) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ac = torch.sqrt(alphas_cumprod)
    sqrt_1mac = torch.sqrt(1.0 - alphas_cumprod)

    pool = torch.nn.AdaptiveAvgPool1d(1).to(dev)
    probes = torch.tensor([50, 250, 500, 750, 950], dtype=torch.long, device=dev)
    print(f"  DiffEEG(noised) params: {sum(p.numel() for p in bb.parameters()):,}; "
          f"sqrt(1-abar) at probes = "
          f"{[round(float(sqrt_1mac[t]),3) for t in probes.tolist()]}", flush=True)

    def enc(x, t):
        te = bb.time_mlp(t)
        h = bb.init_conv(x)
        outs = []
        for ml in bb.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], torch.nn.Conv1d):
                h = ml[0](h)
            else:
                for blk in ml:
                    h = (blk(h, te) if "time_emb" in blk.forward.__code__.co_varnames else blk(h))
                outs.append(pool(h).squeeze(-1))
        return torch.cat(outs, 1)

    @torch.no_grad()
    def run(xb):
        x0 = torch.from_numpy(xb).float().to(dev)
        g = torch.Generator(device=dev).manual_seed(0)   # reproducible corruption
        feats = []
        for ts in probes:
            eps = torch.randn(x0.shape, generator=g, device=dev, dtype=x0.dtype)
            xt = sqrt_ac[ts] * x0 + sqrt_1mac[ts] * eps   # forward diffusion to level t
            feats.append(enc(xt, ts.expand(x0.shape[0])))
        return torch.cat(feats, 1)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["eegdm", "diffeeg", "biot", "labram", "cbramod",
                                        "eegmamba", "diffeeg_noised", "diffeeg_win",
                                        "diffeeg_tok"], required=True)
    ap.add_argument("--dataset", choices=["tuab5s", "tusz", "bonn", "siena"], required=True)
    a = ap.parse_args()

    import data_adapter as A
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    sp_file = B / "results" / f"splits_{a.dataset}.json"
    splits = get_splits(a.dataset, A)
    if not sp_file.exists():
        sp_file.write_text(json.dumps(
            {"dataset": a.dataset, "seed": SEED,
             "caps": {"train": TRAIN_CAP, "val": VAL_CAP, "test": TEST_CAP},
             "n": {k: len(v) for k, v in splits.items()},
             "classes": A.DATASETS[a.dataset]["classes"]}, indent=2))

    EXTRACTORS = {"eegdm": eegdm_extractor, "diffeeg": diffeeg_extractor,
                  "biot": biot_extractor, "labram": labram_extractor,
                  "cbramod": cbramod_extractor, "eegmamba": eegmamba_extractor,
                  "diffeeg_noised": diffeeg_noised_extractor,
                  "diffeeg_win": diffeeg_win_extractor,
                  "diffeeg_tok": diffeeg_tok_extractor}
    # Adding a model means touching four places: CLI choices, BATCH, EXTRACTORS and
    # the view map. Missing one used to surface as a KeyError mid-run after the
    # backbone had already loaded; check them up front instead.
    VIEW = {"diffeeg_noised": "diffeeg", "eegmamba": "cbramod",
            "diffeeg_win": "diffeeg", "diffeeg_tok": "diffeeg"}
    # Token features carry a window axis and are ~8x larger than a pooled vector;
    # fp16 halves the file without mattering to a probe or a small head.
    DTYPE = {"diffeeg_tok": np.float16}
    _registered = set(EXTRACTORS) & set(BATCH)
    _declared = set(ap._actions[1].choices)
    if _declared != _registered:
        raise RuntimeError(
            f"model registration mismatch: CLI declares {sorted(_declared)}, "
            f"extractors+BATCH cover {sorted(_registered)}")

    run = EXTRACTORS[a.model](dev)
    view = VIEW.get(a.model, a.model)
    bs = BATCH[a.model]

    for name, idx in splits.items():
        feats, ys = [], []
        for i in range(0, len(idx), bs):
            xb, yb = A.materialise(idx[i:i + bs], a.dataset, view)
            feats.append(run(xb).detach().cpu().numpy().astype(DTYPE.get(a.model, np.float32)))
            ys.append(yb)
            if i and i % (bs * 200) == 0:
                print(f"    {name} {i}/{len(idx)}", flush=True)
        F, Y = np.concatenate(feats), np.concatenate(ys)
        np.savez_compressed(OUT / f"{a.model}_{a.dataset}_{name}.npz", X=F, y=Y)
        print(f"  {name:<6} {F.shape}  labels {np.bincount(Y)}", flush=True)


if __name__ == "__main__":
    main()
