# Solar Transport Fusion V17

V17 is a from-scratch test of whether the images contain a transferable solar
wind transport signal. It does not use a CNN-LSTM, U-Net, Transformer, or a
learned image gate.

## Architecture

For every one of the 20 images, V17 keeps all 64 native longitude columns. It
computes `mean`, `std`, dark fraction, and bright fraction in north, center,
and south latitude bands for both channels. Signed temporal differences are
added before a small stride-one longitude encoder.

Each time/longitude token predicts a mixture over five fixed speed experts:

```text
300, 400, 500, 650, 800 km/s
```

An expert's speed is both its delivered wind value and the input to its
transit time. Solar longitude adds a rotation wait. Arrival kernels have a
15-hour width and are limited to delays between 48 and 144 hours. The model
therefore cannot use one latent variable for timing and an unrelated variable
for value.

The final prediction starts at a train-only global AR(2) forecast. The
transport-minus-AR difference enters a residual-scale-bounded correction
directly. There is no learned scalar gate that can collapse the complete image
path. Dropping images returns exactly AR(2).

## Training Stages

1. `transport`: image transport reconstructs the last 10 observed wind values
   from causally earlier images. Future target labels are not used.
2. `fusion`: the transport encoder is frozen and only the AR fusion head is
   trained on the 12 future targets.
3. `joint`: both parts are fine-tuned at a lower learning rate with weak
   hindcast and raw-transport objectives.

One command runs three controls:

- `v17_native`: all three stages with native images.
- `v17_scrambled`: identical training after deterministic time/longitude
  reversal.
- `v17_no_pretrain`: native images but no transport pretraining.

The image hypothesis is supported only if `v17_native` beats both controls on
validation RMSE and chain-macro RMSE, while also producing a lower transport
hindcast RMSE than the scrambled run.

## Server Run

```bash
cd ~/solar-wind-taeuk
git switch taeukjung
git pull --ff-only origin taeukjung
mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v17_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_transport_fusion_v17_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Run only the primary model when time is short:

```bash
V17_EXPERIMENTS=v17_native \
nohup bash scripts_taeukjung/run_solar_transport_fusion_v17_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
```

Inference uses the same experiment selection:

```bash
V17_EXPERIMENTS=v17_native \
bash scripts_taeukjung/run_solar_transport_fusion_v17_ablation_server_cuda.sh infer
```

Outputs are written below
`/home/jovyan/outputs/solar_transport_fusion_v17_seed777/`.
