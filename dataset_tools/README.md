# External EEG Dataset Preparation

These helpers convert external EEG datasets into the same structural format used
by the TUAB experiments:

```text
processed_dataset/
  train-normal/*_batch_0000.npy
  train-seizure/*_batch_0000.npy
  test-normal/*_batch_0000.npy
  test-seizure/*_batch_0000.npy
  metadata.json
```

Each batch is `float32` with shape `(N, 22, 2500)`: 5 seconds at 500 Hz in the
22-channel frame expected by the pretrained EEGdiff V2 backbone. Missing
channels are zero-filled and recorded in `metadata.json`.

## Download Sources

Inspect download commands without running them:

```bash
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/dataset_tools/download_public_eeg.py --print
```

Run open downloads into a raw-data root:

```bash
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/dataset_tools/download_public_eeg.py \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw \
  --datasets sleep_edfx siena bonn mindbigdata
```

TUEV requires TUH/NEDC access approval, ssh-key registration, and rsync
credentials. Run the command printed by `--print` after your TUH access works.
For Bern-Barcelona, the script writes a pointer file because the UPF page is the
stable source of the current download link.

## Convert Raw Data

Examples:

```bash
# Siena: seizure windows become positive; non-seizure windows become normal.
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/dataset_tools/prepare_external_eeg.py \
  --dataset siena \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/siena \
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/siena

# Sleep-EDF: writes stage-named folders and a binary wake-vs-sleep view.
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/dataset_tools/prepare_external_eeg.py \
  --dataset sleep_edfx \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/sleep_edfx \
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx

# Bonn: set E is seizure, A-D are normal/control.
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/dataset_tools/prepare_external_eeg.py \
  --dataset bonn \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/bonn \
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bonn
```

## Evaluate a Binary Checkpoint

Use this only for processed folders with binary `normal`/`seizure` or
`background`/`event` labels:

```bash
/home/abdulh/venvs/pt-vae-alliance/bin/python EEGdiff_V2/finetuning/eval_processed_binary.py \
  --processed-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/siena \
  --ckpt /home/abdulh/scratch/EEGdiff_V2/TUAB_Res/run_20260702_034225/unfreeze_3/best.pth
```

