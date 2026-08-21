# Solar Peak Event V15

V15 keeps the V14 U-Net, physics-guided deformable timing, causal masking, and
AR(2) fallback. It adds direct labels that are available from every training
target without external annotation:

- **when**: the 12-bin future index containing the maximum wind speed;
- **peak value**: the maximum normalized wind speed across those 12 bins.

The timing target is softened by one 6-hour bin. Its loss is downweighted when
the target peak has low prominence, because the exact argmax of a nearly flat
sequence is not a meaningful timing label. The value branch is trained with
peak RMSE. Both metrics are written to history in physical units.

The branches cannot replace the complete forecast. They create a Gaussian
event curve around the predicted time and apply a small AR-residual-capped
correction toward the predicted peak value. This path is gated off by image
masking, so modality-drop samples and image-free inference return exactly the
AR forecast.

## Server run

```bash
cd ~/solar-wind-taeuk
git switch taeukjung
git pull --ff-only origin taeukjung

mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v15_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_peak_event_v15_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

The default command runs peak-time weights `0.05` and `0.10`, both with peak
value weight `0.25`. To run only the balanced default:

```bash
V15_EXPERIMENTS=v15_peak_joint \
  bash scripts_taeukjung/run_solar_peak_event_v15_ablation_server_cuda.sh train
```
