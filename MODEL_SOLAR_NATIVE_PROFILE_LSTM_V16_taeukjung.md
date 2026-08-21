# Solar Native Profile LSTM V16

V16 is a deliberately small control model. It never resizes or spatially
downsamples the 64 x 64 images. For each of the 64 original longitude columns,
it calculates disk-masked latitude `mean`, `min`, `max`, and `std` for both
channels. Signed temporal differences double this to 16 features per column.

Two stride-one longitude convolutions keep all 64 columns. A single linear
frame projection and one-layer LSTM feed a fixed 96-hour lag attention. The
train-only AR(2) forecast is the wind baseline; the neural wind residual is
frozen at zero. Images can only add a residual-scale-capped correction, and
image masking returns exactly AR(2).

The default command runs three matched controls: native images, deterministic
time/longitude-scrambled images, and wind-only AR(2). A useful image result
must beat both controls on validation RMSE and chain-macro RMSE.

```bash
cd ~/solar-wind-taeuk
git switch taeukjung
git pull --ff-only origin taeukjung
mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v16_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_native_profile_lstm_v16_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```
