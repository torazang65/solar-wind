# Solar Deformable Timing V14

V14 keeps V13's Lite U-Net source-speed map, train-only AR(2) guard, causal
hindcast queries, and speed-locked transit equation. It replaces dense
cross-attention over all 320 source cells with physics-guided deformable
attention.

For each query and attention head, V14 first selects eight source cells whose
ballistic arrival times are closest to the requested horizon. A learned module
then moves seven references by at most 12 hours and 1.5 longitude cells. The
first reference cannot move, so every set retains a physical anchor. Offsets
are clipped to the query's acquisition-time causal boundary. Latitude is kept
fixed because the PNG data has no direct supervision for a free 2-D warp.

The sampled source speed still has one meaning: it controls transit time and is
the value delivered at arrival. The attention block cannot synthesize an
independent wind value. A differentiable dense reconstruction of the sparse
attention remains available for V13-compatible target backmapping.

## Server ablation

```bash
cd ~/solar-wind-taeuk
git switch taeukjung
git pull --ff-only origin taeukjung

mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v14_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_deformable_timing_v14_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Set `V14_EXPERIMENTS=v14_deformable_full` to run only the default model. The
two-run default compares target backmapping weight `0.01` with `0.0` while
holding architecture, split, and seed fixed.

## Local pipeline check

```bash
conda activate ASAI
bash scripts_taeukjung/run_solar_deformable_timing_v14_local_mps.sh smoke
bash scripts_taeukjung/run_solar_deformable_timing_v14_local_mps.sh train
```
