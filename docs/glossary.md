# EEGdiff / DiffEEG — Glossary of Terms

A reference for the terminology used throughout this project (the DiffEEG paper,
the processing scripts, and the evaluation ladder). Grouped by theme.

---

## 1. Evaluation / adaptation settings

These describe *how much* (if anything) we train on top of the frozen diffusion
backbone. Ordered from "no training" to "most training".

### Zero-shot transfer
Take a classifier that was **already trained on another dataset/task** and apply
it to a **new** dataset with **no further training**.
- Uses **no** labels from the target dataset.
- Only works when the label space matches (e.g. seizure → seizure).
- In our work: reuse the TUSZ-trained seizure head on Bonn / Siena
  (Bonn ROC-AUC 0.716, Siena 0.626).

### Parameter-free / non-parametric probe (k-NN)
Classify each point by its **nearest neighbours** in the frozen embedding space.
A classifier with **no learned weights at all** (nothing is trained).
- Uses the target dataset's **own labels** — the training embeddings form the
  neighbour bank — but performs **no gradient training**.
- **NOT** the same as zero-shot: zero-shot uses no target labels; k-NN does.
- Measures the *local geometry* / clustering of the embedding.
- In our work: 5-fold CV balanced-accuracy k-NN (k=10) on the 4800-d embedding
  (`tab:knn`, and the t-SNE panel annotations).

### Linear probing (a.k.a. head-only)
Freeze the backbone, attach a **new** classifier head, and **train only that
head** on the target dataset's training split. The backbone never updates.
- "Linear" is used loosely: our head is a small **2-layer MLP**
  (4800 → 512 → 256 → K, BatchNorm + GELU + dropout), not a single linear layer.
- Isolates the quality of the frozen representation with a small trained readout.
- In our tables: the "Head only, no RL" rows.

### Partial fine-tuning (unfreezing)
Unfreeze **some** of the backbone (from the deepest encoder level up to the whole
encoder) and train it together with the head, using a gentler learning rate on
the backbone (`η_backbone = 0.01 · η_head`).
- Flags: `--unfreeze last_level | last_two_levels | encoder_all | all`.

### Full fine-tuning
Unfreeze the **entire** backbone (`--unfreeze all`) and train end-to-end.

### RL decision head (REINFORCE)
An add-on classification head trained with a policy-gradient (REINFORCE)
objective whose reward is **macro-F1**, to directly optimise the imbalanced-class
metric rather than cross-entropy. Combined with any of the above (e.g.
"Unfrozen + RL"). Tried PPO as an alternative — did not beat REINFORCE.

> **One-line summary:** zero-shot (reuse a foreign classifier) → k-NN (no
> training, target labels only) → linear probing (train a head) → partial FT →
> full FT (+ optional RL head).

---

## 2. Metrics

### Accuracy (plain)
Fraction of correctly classified samples. **Misleading under class imbalance**
(a majority-only predictor scores high).

### Balanced accuracy
Mean of the **per-class recall** (= macro-averaged recall). Robust to imbalance;
a majority-only predictor scores at chance (= 1 / #classes). This is what
CBraMod / LaBraM / REVE report, and what we report to compare with them.
- **Balanced accuracy (the metric) ≠ balanced dataset (subsampling).** The
  foundation-model baselines use the **full imbalanced data** and report the
  balanced-accuracy metric — they do **not** subsample to balance the data.

### Macro-F1
Unweighted mean of the per-class F1 scores (each class counts equally).
Imbalance-robust; good for multiclass tasks with rare classes.

### Weighted-F1 (W-F1)
Mean of per-class F1 weighted by class support. Closer to plain accuracy;
dominated by the majority class.

### ROC-AUC (AUROC)
Area under the ROC curve (true-positive vs false-positive rate). **Invariant to
class prevalence** — it does not change if you resample the negative class.

### AUC-PR / PR-AUC (Average Precision)
Area under the precision–recall curve. **Strongly depends on prevalence** —
evaluating on a balanced test set inflates it. (This is the confound behind the
"effect of test-set class balance" section: same model, AUPRC rises from 0.48 at
6.7% seizure prevalence to 0.86 at full balance, while AUROC is unchanged.)

### Cohen's Kappa (κ)
Agreement between predictions and labels **corrected for chance**. Used for
multiclass tasks (e.g. TUEV) and inter-rater reliability.

### Sensitivity / Specificity / Precision / Recall
- **Sensitivity** = **Recall** (of the positive class) = TP / (TP + FN) — ability
  to catch positives (e.g. seizures).
- **Specificity** = TN / (TN + FP) — ability to correctly pass normals.
- **Precision** = TP / (TP + FP) — of the predicted positives, how many are real.

### Chance level
The score a trivial/random classifier gets: 1 / #classes for balanced accuracy
and accuracy; 0.5 for ROC-AUC.

---

## 3. Data splits (how train/test are separated)

### Subject-wise / subject-independent split
Each **subject** appears in **only one** of train / test. Measures genuine
generalisation to new people. The honest, harder setting for within-subject
datasets (SEED-V, EEGMMIDB, Mumtaz).

### Patient-wise split
The clinical version of subject-wise: no **patient** appears in both sides
(TUSZ, TUAB, Siena). The official TUH corpus splits are patient-disjoint by
construction.

### Clip-wise / trial-wise split (CBraMod 5:5:5)
Split by **stimulus clip / trial** within each session, so **all subjects appear
in every split**; only the movie clips differ. This is how CBraMod / LaBraM /
REVE evaluate SEED-V and (via EC+EO windows) Mumtaz — subject-*dependent* but
clip-disjoint. Must match this to compare against their leaderboard numbers.

### Pooled random split (leaky / within-subject diagnostic)
Pool all windows and draw a fresh random train/test split, **ignoring** the
official subject split — the same subject (and often the same recording) leaks
into both sides. Inflates results; used only as a diagnostic to show how much of
a gap is due to the split vs the model.

### Prevalence-matched evaluation
Re-score the **same** model probabilities after subsampling the negative class to
a target positive-class fraction (e.g. 6.7% natural vs 29% "balanced" as used by
seizure-FM benchmarks). Reveals how much of an AUPRC gap is a test-set-balance
artefact rather than a real capability difference.

---

## 4. Model & pipeline concepts

### Diffusion backbone
`DeepEnhancedEEGDiffusionModel` — a 1-D U-Net (40.7 M params) pretrained with a
**denoising-diffusion** objective (predict the noise added to EEG at a random
timestep) on non-seizure clinical EEG. No labels are used in pretraining
(self-supervised / generative).

### Multi-timestep diffusion-probing embedding
To turn the generative backbone into a feature extractor: run the frozen encoder
at 5 diffusion timesteps (50, 250, 500, 750, 950), global-average-pool each
encoder level, and concatenate → a fixed **4800-dimensional** embedding
(64·(1+2+4+8)·5), independent of the input length.

### Channel harmonization (22-channel canonical frame)
Map any dataset's electrodes onto our fixed 22-channel 10–20 frame **by name**
(select — never average or interpolate, because the backbone's filters are
electrode-position specific). Missing electrodes are **zero-filled**;
first-match-wins so a real electrode is never overwritten by an alias.
- Canonical order: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5,
  T6, A1, A2, FZ, CZ, PZ, ROC.

### Normalization (own stats, z-scoring)
Per-channel z-scoring using **each dataset's own** mean/std (computed over its
train split). Makes the physical unit irrelevant (volts vs µV) — never reuse
another dataset's stats, which would crush the signal toward zero.

### Windowing
Cut continuous EEG into fixed **5-second / 1280-sample windows at 256 Hz**
(the backbone's native input). Non-overlapping unless a shorter stride is used to
increase sample count.

### Common-average reference (CAR)
Re-reference each electrode by subtracting the mean across the present scalp
electrodes. A standard preprocessing step (used by CBraMod/REVE); we apply it in
the "matched" Mumtaz pipeline.

### Band-pass / notch filter
- **Band-pass** (e.g. 0.1–30 Hz): keep the physiologically informative band,
  remove slow drift and high-frequency EMG/noise.
- **Notch** (50 or 60 Hz): remove power-line interference.

---

## 5. Quick reference — what we report per task type

| Task type | Primary metrics |
|---|---|
| Binary clinical (TUAB, TUSZ, Siena, Bonn, Mumtaz) | Balanced acc, ROC-AUC, AUC-PR |
| Multiclass events / sleep (TUEV, Sleep-EDFx) | Balanced acc, macro-F1, Cohen's κ |
| Motor / emotion (EEGMMIDB, SEED-V) | Balanced acc, macro-F1 |

*Balanced accuracy is the common comparison metric against foundation-model
baselines (CBraMod, LaBraM, REVE, NeurIPT, BIOT).*
