# Isolated residual AE and diffusion stage

This package ports the residual autoencoder and conditional latent diffusion
from `nowcast_gla` without modifying the existing next-frame model or trainer.

The frozen coarse model always rolls forward from historical radar, satellite,
and rain fields. Ground-truth future modalities are used only as targets when
forming `target_rain - coarse_rain`; they are never inserted into rollout input.

The AE checkpoint stores calibrated per-channel latent mean and standard
deviation. Diffusion training always operates on normalized residual latents.
The original v1 6/9 epochs with 1500 training batches per epoch were scaled for the
5426-window multi-month dataset and match the old experiment's approximate AE
and diffusion update counts. Override `--max-train-batches` only when you
intentionally want a shorter diagnostic or a full-dataset pass.

`config_v2.yaml` is the longer residual-learning run. It performs 30 epochs,
1500 batches per epoch, and one optimizer update per four batches: about 11,250
updates versus about 1,688 in v1. In addition to epsilon MSE, it decodes the
one-step x0 estimate and supervises final rainfall L1, active residual pixels,
residual bias, residual standard deviation, and spatial direction/correlation.
Decoded-space terms are weighted by `sqrt(alpha_bar)` so very noisy timesteps do
not dominate them. Checkpoint selection samples 20 validation batches instead
of four.

The completed coarse checkpoint is loaded read-only. With Accelerate automatic
checkpoint naming, use the directory that actually contains `model.safetensors`
(currently `.../checkpoints/checkpoint_12`), not the adjacent directory that
contains only `meta.json`.

Run the AE on the RTX 4090 (physical GPU 1):

```bash
cd /home/rainpred/RainPrediction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/rainpred/miniconda3/envs/rainpred/bin/python \
  -m src.residual_diffusion_stage.train \
  --stage ae \
  --coarse-checkpoint runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12 \
  --output-dir runs/residual_diffusion_cross_local_v1
```

Then train diffusion from the calibrated AE checkpoint:

```bash
cd /home/rainpred/RainPrediction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/rainpred/miniconda3/envs/rainpred/bin/python \
  -m src.residual_diffusion_stage.train \
  --stage diffusion \
  --coarse-checkpoint runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12 \
  --ae-checkpoint runs/residual_diffusion_cross_local_v1/ae/final.pt \
  --output-dir runs/residual_diffusion_cross_local_v1
```

For the longer v2 run, reuse the already validated v1 AE and write to a new
directory:

```bash
cd /home/rainpred/RainPrediction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/rainpred/miniconda3/envs/rainpred/bin/python \
  -m src.residual_diffusion_stage.train \
  --stage diffusion \
  --stage-config src/residual_diffusion_stage/config_v2.yaml \
  --coarse-checkpoint runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12 \
  --ae-checkpoint runs/residual_diffusion_cross_local_v1/ae/final.pt \
  --diffusion-checkpoint runs/residual_diffusion_cross_local_v1/diffusion/best.pt \
  --output-dir runs/residual_diffusion_cross_local_v2
```

`CUDA_VISIBLE_DEVICES=1` exposes the physical RTX 4090 as logical `cuda:0`
inside the process. The two commands never update the coarse model and never
insert ground-truth future radar or satellite fields into its rollout history.
Diffusion validation logs coarse/final L1 and CSI at 0.1, 0.3, and 0.5 using
the same thresholds as the coarse trainer. `best.pt` is selected by final L1.

After diffusion training, run the residual-scale ablation. Every scale reuses
the same coarse forecast and the same four-member diffusion ensemble:

```bash
cd /home/rainpred/RainPrediction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/rainpred/miniconda3/envs/rainpred/bin/python \
  -m src.residual_diffusion_stage.ablate_residual_scale \
  --coarse-checkpoint runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12 \
  --ae-checkpoint runs/residual_diffusion_cross_local_v1/ae/final.pt \
  --diffusion-checkpoint runs/residual_diffusion_cross_local_v1/diffusion/best.pt \
  --output-dir runs/residual_diffusion_cross_local_v1
```

The full report is written to
`runs/residual_diffusion_cross_local_v1/ablation/residual_scale_ablation.json`.
It includes overall and per-lead MAE, RMSE, bias, CSI, POD, FAR, and HSS. The
recommended scale maximizes mean CSI among candidates whose MAE is no worse
than the scale-zero coarse baseline. The report also lists the best scale for
each individual metric.
