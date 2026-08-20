# Taeuk Jung workspace

This branch keeps Taeuk's implementation separate from the shared baseline.

## Layout

- `src/`: shared baseline
- `src_taeukjung/`: mask and soft-cubic preprocessing implementation
- `scripts_taeukjung/run_preprocess_64_20epoch.py`: fixed four-way, 64 px, 20-epoch ablation runner

## Server workflow

Run the personal branch directly while it is under development:

```bash
git fetch origin
git switch taeukjung
git pull --ff-only origin taeukjung
```

Run the focused training configuration:

```bash
IMAGE_SIZE=64 \
EPOCHS=20 \
BATCH_SIZE=256 \
NUM_WORKERS=4 \
SOLAR_DISK_MASK=1 \
IMAGE_NORM=soft_cubic \
SOFT_CUBIC_STRENGTH=0.25 \
OUTPUT_DIR=outputs/baseline_6h_taeukjung \
python src_taeukjung/train.py
```

Run the four-way ablation:

```bash
python scripts_taeukjung/run_preprocess_64_20epoch.py \
  --image-size 64 \
  --epochs 20 \
  --batch-size 256 \
  --num-workers 4 \
  --run-name ablation_4way_64_20
```

Merge into `main` only after the branch result has been reviewed.
