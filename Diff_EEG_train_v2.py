#!/usr/bin/env python3
import os
import argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegdiff_v2_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/eegdiff_v2_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import torch.optim as optim
from tqdm import tqdm
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import time

# ------------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------------
DEFAULT_THUSZ_NON_SEIZURE_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/Pretraining_data/non_seizure_batches"
DEFAULT_CHBMIT_NON_SEIZURE_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/CHBMIT/processed_chbmit/non-seizure"
DEFAULT_TUAB_TRAIN_NORMAL_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/processed_tuab/train-normal"
DEFAULT_TUAB_TEST_NORMAL_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/processed_tuab/test-normal"
DEFAULT_OUTPUT_DIR = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2"


@dataclass(frozen=True)
class PretrainSource:
    name: str
    path: str
    pattern: str = "*.npy"
    train_fraction: float = 0.9


MODEL_CONFIGS = {
    "base": {
        "model_channels": 32,
        "channel_multipliers": [1, 2, 4, 8],
        "num_res_blocks": 2,
        "time_emb_dim": 512,
        "dropout": 0.1,
        "attention_heads": 8,
    },
    "large": {
        "model_channels": 64,
        "channel_multipliers": [1, 2, 4, 8],
        "num_res_blocks": 3,
        "time_emb_dim": 768,
        "dropout": 0.1,
        "attention_heads": 8,
    },
    "xlarge": {
        "model_channels": 64,
        "channel_multipliers": [1, 2, 4, 8, 8],
        "num_res_blocks": 3,
        "time_emb_dim": 1024,
        "dropout": 0.1,
        "attention_heads": 8,
    },
}


def is_data_batch_file(path: Path) -> bool:
    return path.suffix == ".npy" and not path.name.endswith(("_labels.npy", "_pids.npy", "_files.npy"))


def load_npy_shape(path: Path) -> Tuple[int, ...]:
    with open(path, "rb") as fp:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fp)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(fp)
    return tuple(shape)


def discover_batch_files(source: PretrainSource) -> List[Path]:
    source_dir = Path(source.path)
    files = sorted(path for path in source_dir.glob(source.pattern) if is_data_batch_file(path))
    if not files:
        raise ValueError(f"No data batch files found for source '{source.name}' in {source_dir}")
    return files


def split_source_files(files: List[Path], train_fraction: float) -> Tuple[List[Path], List[Path]]:
    split_idx = int(train_fraction * len(files))
    if len(files) > 1:
        split_idx = min(max(split_idx, 1), len(files) - 1)
    return files[:split_idx], files[split_idx:]


def compute_combined_normalization_stats(
    batch_files: List[Path],
    max_samples: int = 50000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel normalization from a mixed multi-dataset corpus."""
    rng = np.random.default_rng(seed)
    file_shapes = [(path, load_npy_shape(path)) for path in batch_files]
    total_samples = sum(shape[0] for _, shape in file_shapes)
    samples_to_use = min(max_samples, total_samples)
    if samples_to_use <= 0:
        raise ValueError("Cannot compute normalization stats with zero samples")

    probabilities = np.asarray([shape[0] for _, shape in file_shapes], dtype=np.float64)
    probabilities /= probabilities.sum()
    chosen_files = rng.choice(len(file_shapes), size=samples_to_use, replace=True, p=probabilities)

    per_file_counts: Dict[Path, int] = {}
    for file_idx in chosen_files:
        path = file_shapes[file_idx][0]
        per_file_counts[path] = per_file_counts.get(path, 0) + 1

    sum_ = None
    sum_sq = None
    count = 0
    for path, n_samples in tqdm(per_file_counts.items(), desc="Computing combined stats"):
        data = np.load(path, mmap_mode="r", allow_pickle=True)
        indices = rng.integers(0, data.shape[0], size=n_samples)
        chunk = np.asarray(data[indices], dtype=np.float32)
        if sum_ is None:
            n_channels = chunk.shape[1]
            sum_ = np.zeros((n_channels, 1), dtype=np.float64)
            sum_sq = np.zeros((n_channels, 1), dtype=np.float64)
        sum_ += np.mean(chunk, axis=(0, 2))[:, np.newaxis] * len(chunk)
        sum_sq += np.mean(chunk ** 2, axis=(0, 2))[:, np.newaxis] * len(chunk)
        count += len(chunk)

    if sum_ is None or sum_sq is None or count == 0:
        raise ValueError("Failed to compute normalization stats")

    mean = sum_ / count
    std = np.sqrt((sum_sq / count) - (mean ** 2)) + 1e-6
    std_cap = np.percentile(std.flatten(), 95)
    std = np.clip(std, 0.1, std_cap)
    print(f"Combined mean range: [{mean.min():.3f}, {mean.max():.3f}], std range: [{std.min():.3f}, {std.max():.3f}]")
    return mean.astype(np.float32), std.astype(np.float32)


def build_pretrain_sources(args: argparse.Namespace) -> List[PretrainSource]:
    sources = [
        PretrainSource("THUSZ_non_seizure", args.thusz_non_seizure_dir, "non_seizure_batch_*.npy"),
        PretrainSource("CHBMIT_non_seizure", args.chbmit_non_seizure_dir, "non-seizure_batch_*.npy"),
        PretrainSource("TUAB_train_normal", args.tuab_train_normal_dir, "normal_batch_*.npy"),
        PretrainSource("TUAB_test_normal", args.tuab_test_normal_dir, "normal_batch_*.npy"),
    ]
    return sources


def prepare_multi_source_splits(
    sources: List[PretrainSource],
) -> Tuple[List[Path], List[Path], List[dict]]:
    train_files: List[Path] = []
    val_files: List[Path] = []
    source_report: List[dict] = []
    for source in sources:
        files = discover_batch_files(source)
        source_train, source_val = split_source_files(files, source.train_fraction)
        train_files.extend(source_train)
        val_files.extend(source_val)
        source_report.append({
            **asdict(source),
            "num_files": len(files),
            "train_files": len(source_train),
            "val_files": len(source_val),
        })
        print(f"{source.name}: {len(source_train)} train files, {len(source_val)} val files")
    if not train_files or not val_files:
        raise ValueError("Need at least one train and one validation file")
    return train_files, val_files, source_report


# ------------------------------------------------------------------
# 1. ENHANCED DIFFUSION SCHEDULER
# ------------------------------------------------------------------
class ImprovedDiffusionScheduler:
    def __init__(self, timesteps=1000, beta_schedule='linear', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def sample_random_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,), device=self.device).long()
    
    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# ------------------------------------------------------------------
# 2. SINUSOIDAL POSITION EMBEDDINGS
# ------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

# ------------------------------------------------------------------
# 3. ADVANCED RESIDUAL BLOCK WITH MULTI-SCALE ATTENTION
# ------------------------------------------------------------------
class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, use_attention=True, attention_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Time embedding with better scaling
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.Dropout(dropout)
        )
        
        # Multi-scale normalization
        self.norm1 = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Advanced attention mechanism
        if use_attention:
            self.attn_norm = nn.InstanceNorm1d(out_channels, affine=True)
            self.attn_q = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_k = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_v = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_proj = nn.Conv1d(out_channels, out_channels, 1)
            # Multi-head attention inspired approach
            self.attention_heads = attention_heads
            self.head_dim = out_channels // attention_heads
            assert self.head_dim * attention_heads == out_channels, "out_channels must be divisible by attention_heads"
        
    def forward(self, x, time_emb):
        # First normalization and conv
        B, C, L = x.shape
        h = self.norm1(x)
        
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time embedding conditioning
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.reshape(time_emb.shape[0], -1, 1)
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        # Second normalization and conv
        h = self.norm2(h)
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Self-attention block (multi-scale)
        if self.use_attention:
            attn_input = self.attn_norm(h)
            
            # Compute query, key, value
            q = self.attn_q(attn_input).view(B, self.attention_heads, self.head_dim, L)
            k = self.attn_k(attn_input).view(B, self.attention_heads, self.head_dim, L)
            v = self.attn_v(attn_input).view(B, self.attention_heads, self.head_dim, L)
            
            # Temporal attention: attend over sequence positions at the current resolution.
            attn_weights = torch.einsum('bhcl,bhcm->bhlm', q, k) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1) + 1e-6  # Add epsilon for numerical stability
            
            attn_output = torch.einsum('bhlm,bhcm->bhcl', attn_weights, v)
            attn_output = attn_output.view(B, -1, L)  # Merge heads
            
            # Project back to original dimensions
            h = h + self.attn_proj(attn_output)
        
        # Residual connection
        residual = self.residual_conv(x)
        return h + residual

# ------------------------------------------------------------------
# 4. DEEP ENHANCED EEG DIFFUSION MODEL
# ------------------------------------------------------------------
class DeepEnhancedEEGDiffusionModel(nn.Module):
    def __init__(self, in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                 num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attention_heads = attention_heads
        self.num_levels = len(channel_multipliers)
        
        # Enhanced time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Initial convolution with better initialization
        self.init_conv = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.init_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # ===== ENCODER =====
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        
        # Build encoder with increased depth
        for i, mult in enumerate(channel_multipliers):
            out_channels = model_channels * mult
            
            # Residual blocks at this resolution
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # Place attention in later levels and at the end
                use_attention = (i >= len(channel_multipliers)//2) and (j == num_res_blocks - 1)
                block = AdvancedResidualBlock(
                    current_channels, out_channels, time_emb_dim, dropout, 
                    use_attention=use_attention, attention_heads=attention_heads
                )
                level_blocks.append(block)
                current_channels = out_channels
            
            self.down_blocks.append(level_blocks)
            
            # Downsampling layer (except at last level)
            if i != len(channel_multipliers) - 1:
                downsample = nn.Conv1d(current_channels, current_channels, 3, stride=2, padding=1)
                nn.init.kaiming_normal_(downsample.weight, mode='fan_out', nonlinearity='relu')
                self.down_blocks.append(nn.ModuleList([downsample]))
        
        # ===== BOTTLENECK =====
        # Deeper bottleneck with multiple layers
        self.bottleneck_blocks = nn.ModuleList([
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads),
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads),
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads)
        ])
        
        # ===== DECODER =====
        self.up_blocks = nn.ModuleList()
        current_channels = model_channels * channel_multipliers[-1]  # Start with bottleneck channels
        
        # Build decoder with more sophisticated structure
        for i in range(len(channel_multipliers)):
            level_blocks = nn.ModuleList()
            mult = channel_multipliers[-(i+1)]
            out_channels = model_channels * mult
            
            # Upsampling layer (except at first level)
            if i > 0:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(current_channels, out_channels, 3, padding=1)
                )
                nn.init.kaiming_normal_(upsample[1].weight, mode='fan_out', nonlinearity='relu')
                level_blocks.append(upsample)
                current_channels = out_channels
            
            # Residual blocks at this resolution
            for j in range(num_res_blocks + 1):
                # All blocks take current_channels as input
                block_in_channels = current_channels
                
                # Place attention in earlier decoder levels
                use_attention = (i < len(channel_multipliers)//2) and (j == 0)
                block = AdvancedResidualBlock(
                    block_in_channels, out_channels, time_emb_dim, dropout,
                    use_attention=use_attention, attention_heads=attention_heads
                )
                level_blocks.append(block)
                current_channels = out_channels
            
            self.up_blocks.append(level_blocks)
        
        # Final layers with better initialization
        self.final_norm = nn.InstanceNorm1d(current_channels, affine=True)
        self.channel_fix_conv = nn.Conv1d(current_channels, model_channels, 1)  # Project to fixed channels
        self.final_conv = nn.Conv1d(model_channels, in_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        h = self.init_conv(x)
        
        # ===== ENCODER =====
        skips = []
        
        # Process encoder blocks
        for module_list in self.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                # This is a downsampling layer
                h = module_list[0](h)
            else:
                # Process residual blocks in this level
                for block in module_list:
                    if isinstance(block, AdvancedResidualBlock):
                        h = block(h, t_emb)
                        # Store skip connection from the last residual block of each level
                        if block == module_list[-1]:
                            skips.append(h)
                    else:
                        # This is a downsampling layer
                        h = block(h)
        
        # ===== BOTTLENECK =====
        for block in self.bottleneck_blocks:
            h = block(h, t_emb)
        
        # ===== DECODER =====
        skip_idx = len(skips) - 1
        
        # Process decoder blocks
        for module_list in self.up_blocks:
            skip_applied = False
            for block in module_list:
                if isinstance(block, nn.Sequential) and hasattr(block[0], 'scale_factor'):
                    # Upsampling layer
                    h = block(h)
                else:  # Residual block
                    if not skip_applied and skip_idx >= 0:
                        skip_tensor = skips[skip_idx]
                        skip_idx -= 1
                        if skip_tensor.shape[-1] != h.shape[-1]:
                            skip_tensor = F.interpolate(skip_tensor, size=h.shape[-1], mode='linear', align_corners=False)
                        h = h + skip_tensor
                        skip_applied = True
                    h = block(h, t_emb)
        
        # ===== FINAL LAYERS =====
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.channel_fix_conv(h)
        h = F.silu(h)
        return self.final_conv(h)

# ------------------------------------------------------------------
# 5. DATA LOADING
# ------------------------------------------------------------------
class BatchEEGDataset(Dataset):
    def __init__(
        self,
        batch_dir,
        seq_length=1280,
        normalize=True,
        max_batches=None,
        cache_in_memory=True,
        batch_files=None,
        normalization_stats=None,
        save_normalization=True,
        normalization_dir=None,
    ):
        self.batch_dir = Path(batch_dir)
        self.seq_length = seq_length
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory
        self.save_normalization = save_normalization
        self.normalization_dir = Path(normalization_dir) if normalization_dir else None

        if batch_files is not None:
            self.batch_files = [Path(p) for p in batch_files]
        else:
            # Match both non_seizure_batch_*.npy and seizure_batch_*.npy,
            # but exclude sidecar files.
            self.batch_files = sorted(
                f for f in self.batch_dir.glob("*.npy")
                if is_data_batch_file(f)
            )
        if max_batches:
            self.batch_files = self.batch_files[:max_batches]
        
        if not self.batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        print(f"Found {len(self.batch_files)} batch files")
        
        self.sample_indices = []
        self.batch_data = []
        self._array_cache = {}
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            if self.cache_in_memory:
                data = np.load(batch_file, allow_pickle=True)
                self.batch_data.append(data)
                batch_size = len(data)
            else:
                self.batch_data.append(batch_file)
                with open(batch_file, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    else:
                        shape, _, _ = np.lib.format.read_array_header_2_0(f)
                    batch_size = shape[0]
            
            for sample_idx in range(batch_size):
                self.sample_indices.append((batch_idx, sample_idx))
        
        print(f"Total samples: {len(self.sample_indices)}")
        
        if self.normalize:
            if normalization_stats is None:
                self.compute_normalization_stats()
                if self.save_normalization:
                    norm_dir = self.normalization_dir or (self.batch_dir / "normalization")
                    norm_dir.mkdir(exist_ok=True)
                    np.save(norm_dir / "mean.npy", self.mean)
                    np.save(norm_dir / "std.npy", self.std)
                    print(f"Saved normalization stats to {norm_dir}")
            else:
                provided_mean, provided_std = normalization_stats
                self.mean = np.array(provided_mean)
                self.std = np.array(provided_std)

            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32)

    def _get_batch_array(self, batch_idx):
        batch_ref = self.batch_data[batch_idx]
        if self.cache_in_memory:
            return batch_ref

        cached = self._array_cache.get(batch_idx)
        if cached is None:
            cached = np.load(batch_ref, mmap_mode="r", allow_pickle=True)
            self._array_cache[batch_idx] = cached
        return cached
        
    def compute_normalization_stats(self, num_samples=None, chunk_size=1000):
        # Use all samples for normalization if num_samples is None
        total_samples = len(self.sample_indices)
        if num_samples is None or num_samples > total_samples:
            num_samples = total_samples
        sample_indices = np.arange(total_samples) if num_samples == total_samples else np.random.choice(total_samples, num_samples, replace=False)

        # Chunked computation
        n_channels = None
        n_seq = None
        sum_ = None
        sum_sq = None
        count = 0
        for i in tqdm(range(0, len(sample_indices), chunk_size), desc="Computing stats"):
            chunk_idxs = sample_indices[i:i+chunk_size]
            chunk_samples = []
            for idx in chunk_idxs:
                batch_idx, sample_idx = self.sample_indices[idx]
                batch_arr = self._get_batch_array(batch_idx)
                sample = batch_arr[sample_idx]
                chunk_samples.append(sample)
            chunk_samples = np.stack(chunk_samples)
            if n_channels is None:
                n_channels = chunk_samples.shape[1]
                n_seq = chunk_samples.shape[2]
                sum_ = np.zeros((n_channels, 1), dtype=np.float64)
                sum_sq = np.zeros((n_channels, 1), dtype=np.float64)
            # Compute sum and sum of squares for this chunk
            sum_ += np.mean(chunk_samples, axis=(0, 2))[:, np.newaxis] * len(chunk_samples)
            sum_sq += np.mean(chunk_samples ** 2, axis=(0, 2))[:, np.newaxis] * len(chunk_samples)
            count += len(chunk_samples)
        self.mean = sum_ / count
        self.std = np.sqrt((sum_sq / count) - (self.mean ** 2)) + 1e-6
        std_cap = np.percentile(self.std.flatten(), 95)
        self.std = np.clip(self.std, 0.1, std_cap)
        print(f"Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}], Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")
        
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        batch_idx, sample_idx = self.sample_indices[idx]
        sample = self._get_batch_array(batch_idx)[sample_idx]
        
        if sample.shape[1] != self.seq_length:
            if sample.shape[1] > self.seq_length:
                start = np.random.randint(0, sample.shape[1] - self.seq_length)
                sample = sample[:, start:start + self.seq_length]
            else:
                pad_width = self.seq_length - sample.shape[1]
                sample = np.pad(sample, ((0, 0), (0, pad_width)), mode='edge')
        
        sample = torch.as_tensor(np.asarray(sample, dtype=np.float32))
        if self.normalize:
            sample = (sample - self.mean_tensor) / self.std_tensor
        
        return {"eeg": sample}

    def get_normalization_stats(self):
        if not self.normalize:
            return None, None
        return self.mean.copy(), self.std.copy()

# ------------------------------------------------------------------
# 6. ENHANCED TRAINER WITH BETTER CONFIGURATION
# ------------------------------------------------------------------
class EnhancedDiffusionTrainer:
    def __init__(
        self,
        model,
        diffusion_scheduler,
        train_loader,
        val_loader,
        device,
        lr=5e-5,
        use_mixed_precision=True,
        recon_weight=0.1,
        output_dir=DEFAULT_OUTPUT_DIR,
        run_config=None,
    ):
        
        self.model = model.to(device)
        self.diffusion = diffusion_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.recon_weight = recon_weight
        
        # Much more conservative optimizer settings
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))
        # Better learning rate scheduling
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5000, T_mult=2, eta_min=1e-7)
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
        
        self.step = 0
        self.best_val_loss = float('inf')
        self.output_dir = Path(output_dir)
        self.run_config = run_config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_loss(self, x, t, noise, pred_noise):
        # Clip predictions to prevent extreme values
        pred_noise = torch.clamp(pred_noise, -10, 10)
        
        noise_loss = F.mse_loss(pred_noise, noise)
        
        alpha = self.diffusion.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sigma = self.diffusion.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_recon = (x - sigma * pred_noise) / alpha
        x_recon = torch.clamp(x_recon, -3, 3)
        target_recon = (x - sigma * noise) / alpha
        target_recon = torch.clamp(target_recon, -3, 3)
        recon_loss = F.mse_loss(x_recon, target_recon)
        
        total_loss = noise_loss + self.recon_weight * recon_loss
        return total_loss, {'noise_loss': noise_loss.item(), 'recon_loss': recon_loss.item()}
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_noise_loss = total_recon_loss = 0
        num_batches = 0
        last_progress_log = time.monotonic()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            x = batch["eeg"].to(self.device)
            batch_size = x.shape[0]
            t = self.diffusion.sample_random_timesteps(batch_size)
            noise = torch.randn_like(x)
            noisy_x, true_noise = self.diffusion.add_noise(x, t, noise)
            
            if self.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_noise = self.model(noisy_x, t)
                    loss, loss_components = self.compute_loss(noisy_x, t, true_noise, pred_noise)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.step += 1
                    continue

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_noise = self.model(noisy_x, t)
                loss, loss_components = self.compute_loss(noisy_x, t, true_noise, pred_noise)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.step += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            total_noise_loss += loss_components['noise_loss']
            total_recon_loss += loss_components['recon_loss']
            num_batches += 1
            self.step += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})
            if num_batches <= 5 or time.monotonic() - last_progress_log >= 300:
                print(
                    f"Epoch {epoch} batch {num_batches}/{len(self.train_loader)} "
                    f"loss={loss.item():.6f} lr={self.optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                last_progress_log = time.monotonic()

        if num_batches == 0:
            print(f"Warning: every batch in epoch {epoch} produced NaN/Inf loss; skipping epoch.", flush=True)
            return {'loss': float('nan'), 'noise_loss': float('nan'), 'recon_loss': float('nan')}
        return {'loss': total_loss / num_batches, 'noise_loss': total_noise_loss / num_batches, 'recon_loss': total_recon_loss / num_batches}
    
    @torch.no_grad()
    def validate(self, epoch=None):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            x = batch["eeg"].to(self.device)
            batch_size = x.shape[0]

            # Test on more timesteps for better validation
            test_timesteps = [100, 250, 500, 750, 999]
            for t_val in test_timesteps:
                t = torch.full((batch_size,), t_val, device=self.device).long()
                noise = torch.randn_like(x)
                noisy_x, true_noise = self.diffusion.add_noise(x, t, noise)
                pred_noise = self.model(noisy_x, t)
                loss = F.mse_loss(pred_noise, true_noise)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            print("Warning: every validation batch produced NaN/Inf loss; skipping validation.", flush=True)
            return {'loss': float('nan')}

        avg_loss = total_loss / num_batches
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch=epoch, best=True)

        return {'loss': avg_loss}
    
    def save_checkpoint(self, epoch=None, best=False):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            'epoch': epoch, 'step': self.step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'run_config': self.run_config,
        }
        filename = self.output_dir / ("best_EEGDIFF_V2.pth" if best else "latest_checkpoint.pth")
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")

# ------------------------------------------------------------------
# 7. MAIN TRAINING
# ------------------------------------------------------------------
def train_enhanced_diffusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = build_pretrain_sources(args)
    train_batches, val_batches, source_report = prepare_multi_source_splits(sources)

    print(f"Training batch files: {len(train_batches)}, Validation batch files: {len(val_batches)}")

    norm_dir = output_dir / "normalization"
    norm_dir.mkdir(parents=True, exist_ok=True)
    mean_path = norm_dir / "mean.npy"
    std_path = norm_dir / "std.npy"
    if args.recompute_norm or not (mean_path.exists() and std_path.exists()):
        train_mean, train_std = compute_combined_normalization_stats(
            train_batches,
            max_samples=args.norm_samples,
            seed=args.seed,
        )
        np.save(mean_path, train_mean, allow_pickle=False)
        np.save(std_path, train_std, allow_pickle=False)
        print(f"Saved combined normalization stats to {norm_dir}")
    else:
        train_mean = np.load(mean_path)
        train_std = np.load(std_path)
        print(f"Loaded combined normalization stats from {norm_dir}")

    train_ds = BatchEEGDataset(
        output_dir,
        seq_length=1280,
        normalize=True,
        cache_in_memory=args.cache_in_memory,
        batch_files=train_batches,
        normalization_stats=(train_mean, train_std),
        save_normalization=False,
    )
    val_ds = BatchEEGDataset(
        output_dir,
        seq_length=1280,
        normalize=True,
        cache_in_memory=False,
        batch_files=val_batches,
        normalization_stats=(train_mean, train_std),
        save_normalization=False,
    )
    
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    
    diffusion_scheduler = ImprovedDiffusionScheduler(timesteps=1000, beta_schedule='cosine', device=device)
    
    model_config = MODEL_CONFIGS[args.model_size].copy()
    model = DeepEnhancedEEGDiffusionModel(
        in_channels=22,
        **model_config,
    )
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    run_config = {
        "script": "EEGdiff_V2/Diff_EEG_train_v2.py",
        "model_size": args.model_size,
        "model_config": model_config,
        "sources": source_report,
        "train_batch_files": len(train_batches),
        "val_batch_files": len(val_batches),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "normalization_dir": str(norm_dir),
        "pretraining_scope": "non_seizure_or_normal_only",
        "data_parallel": args.data_parallel,
    }
    with open(output_dir / "run_config.json", "w") as fp:
        json.dump(run_config, fp, indent=2)
    
    trainer = EnhancedDiffusionTrainer(
        model,
        diffusion_scheduler,
        train_loader,
        val_loader,
        device,
        lr=args.lr,
        use_mixed_precision=True,
        recon_weight=0.05,
        output_dir=output_dir,
        run_config=run_config,
    )
    
    # Resume support: determine starting epoch and optionally load checkpoint
    start_epoch = 0
    ckpt_path = None
    if getattr(args, 'resume', None):
        ckpt_path = Path(args.resume)
    else:
        latest_path = output_dir / "latest_checkpoint.pth"
        best_path = output_dir / "best_EEGDIFF_V2.pth"
        if latest_path.exists():
            ckpt_path = latest_path
        elif best_path.exists():
            ckpt_path = best_path

    if ckpt_path and ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_to_load = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: failed to strictly load model state_dict: {e}")
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: failed to load optimizer state: {e}")
        # Always reset scheduler so T_0/T_mult from code (not checkpoint) take effect
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=5000, T_mult=2, eta_min=1e-7
        )
        trainer.step = checkpoint.get('step', trainer.step)
        trainer.best_val_loss = checkpoint.get('best_val_loss', trainer.best_val_loss)
        # Guard against checkpoints that saved 'epoch' as None (e.g. best checkpoints saved from validate())
        start_epoch = int(checkpoint.get('epoch') or 0) + 1
        print(f"Resuming training from epoch {start_epoch}")

    print("\nStarting training...\n" + "=" * 60)

    history = {'train_loss': [], 'train_noise_loss': [], 'train_recon_loss': [], 'val_loss': [], 'learning_rate': []}
    
    # Train longer with better stopping criteria
    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate(epoch) if epoch % args.val_every == 0 else {'loss': history['val_loss'][-1] if history['val_loss'] else 0}
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_noise_loss'].append(train_metrics['noise_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch:04d}: Train={train_metrics['loss']:.6f}, Val={val_metrics['loss']:.6f}, Best Val={trainer.best_val_loss:.6f}")

        import math
        if math.isnan(train_metrics['loss']) or math.isinf(train_metrics['loss']):
            print(f"NaN/Inf loss detected at epoch {epoch}, stopping training.")
            break

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(epoch=epoch, best=False)
        
        try:
            with open(trainer.output_dir / "training_history.json", 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
        except:
            pass
        
        if epoch % args.plot_every == 0:
            plot_training_progress(history, trainer.output_dir)
    
    print("\n" + "=" * 60 + "\nTraining completed!")
    return trainer, history

def plot_training_progress(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(len(history['train_loss']))
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', alpha=0.7)
    axes[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_noise_loss'], label='Noise', alpha=0.7)
    axes[0, 1].plot(epochs, history['train_recon_loss'], label='Recon', alpha=0.7)
    axes[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_loss'], color='orange')
    axes[1, 0].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['learning_rate'], color='purple')
    axes[1, 1].set(xlabel='Epoch', ylabel='LR', title='Learning Rate', yscale='log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_model_with_dummy_data(model_size="large"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model on {device}...\n")
    
    diffusion = ImprovedDiffusionScheduler(timesteps=1000, device=device)
    model_config = MODEL_CONFIGS[model_size].copy()
    model = DeepEnhancedEEGDiffusionModel(
        in_channels=22,
        **model_config,
    ).to(device)
    
    batch_size, seq_length = 4, 1280  # Reduced for testing
    dummy_input = torch.randn(batch_size, 22, seq_length, device=device)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input, dummy_t)
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")
        
        noisy_x, noise = diffusion.add_noise(dummy_input, dummy_t)
        pred_noise = model(noisy_x, dummy_t)
        loss = F.mse_loss(pred_noise, noise)
        print(f"Loss: {loss.item():.4f}")
        print("✓ Training step successful!")
        print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diff-EEG V2 pretraining on larger normal/non-seizure THUSZ, CHB-MIT, and TUAB batches."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="large")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm-samples", type=int, default=50000)
    parser.add_argument("--recompute-norm", action="store_true")
    parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (default: pick latest in output dir)")

    parser.add_argument("--thusz-non-seizure-dir", default=DEFAULT_THUSZ_NON_SEIZURE_DIR)
    parser.add_argument("--chbmit-non-seizure-dir", default=DEFAULT_CHBMIT_NON_SEIZURE_DIR)
    parser.add_argument("--tuab-train-normal-dir", default=DEFAULT_TUAB_TRAIN_NORMAL_DIR)
    parser.add_argument("--tuab-test-normal-dir", default=DEFAULT_TUAB_TEST_NORMAL_DIR)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.val_every <= 0 or args.save_every <= 0 or args.plot_every <= 0:
        parser.error("--val-every, --save-every, and --plot-every must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.skip_smoke_test or test_model_with_dummy_data(args.model_size):
        print("\n" + "="*60)
        print("Model test passed! Starting training...")
        print("="*60 + "\n")
        
        try:
            trainer, history = train_enhanced_diffusion(args)
            print("\nTraining completed successfully!")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nModel test failed! Fix the architecture issues first.")
