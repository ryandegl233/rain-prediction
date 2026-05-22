#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import accelerate
import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.metrics.compute_metrics_new import RainGlobalMetricsAccumulator

class StrictMapDataset(Dataset):
    """Adapter for LitData / WIDS / generic datasets."""

    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        if hasattr(self.ds, "valid_indices"):
            return len(self.ds.valid_indices)
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class StrictClassMetricsAccumulator:
    """Strict class equality metrics (non-cumulative)."""

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.hits = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.misses = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.false_alarms = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.correct_negatives = torch.zeros(num_classes, dtype=torch.long, device=device)

    def update(self, pred, gt):
        for c in range(self.num_classes):
            p_c = pred == c
            g_c = gt == c

            self.hits[c] += (p_c & g_c).sum()
            self.misses[c] += (~p_c & g_c).sum()
            self.false_alarms[c] += (p_c & ~g_c).sum()
            self.correct_negatives[c] += (~p_c & ~g_c).sum()

    def compute(self):
        res = {}
        for c in range(self.num_classes):
            h = self.hits[c].float()
            m = self.misses[c].float()
            fa = self.false_alarms[c].float()
            cn = self.correct_negatives[c].float()
            total = h + m + fa + cn

            csi = h / (h + m + fa + 1e-8)
            pod = h / (h + m + 1e-8)
            far = fa / (h + fa + 1e-8)
            f1 = 2 * h / (2 * h + m + fa + 1e-8)

            expected_correct = ((h + m) * (h + fa) + (cn + m) * (cn + fa)) / (total + 1e-8)
            hss = (h + cn - expected_correct) / (total - expected_correct + 1e-8)

            res[f"Class_{c}"] = {"CSI": csi, "POD": pod, "FAR": far, "F1": f1, "HSS": hss}
        return res


class RainPredictionTester:
    def __init__(self, cfg):
        self.cfg = cfg

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(2025)
        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)

        # Use the dataset-provided loader (StreamingDataLoader for LitData) first.
        # Re-wrapping this iterable dataset with a plain torch DataLoader can make
        # worker-sharding semantics inconsistent under Accelerate.
        self.test_dataset, clean_loader = hydra.utils.instantiate(cfg.dataset.val)
        self._fail_fast_if_empty_dataset()
        if clean_loader is None:
            strict_dataset = StrictMapDataset(self.test_dataset)
            clean_loader = DataLoader(
                strict_dataset,
                batch_size=cfg.dataset.val.get("batch_size", 8),
                shuffle=False,
                num_workers=cfg.dataset.val.get("num_workers", 6),
                pin_memory=True,
                drop_last=False,
            )

        self.raw_test_loader = clean_loader
        self._run_diagnostics_before_accelerate()
        self.test_dataloader = self.accelerator.prepare(clean_loader)
        self.model = hydra.utils.instantiate(cfg.rain_prediction_model)
        self.model.to(self.device)
        self.model.eval()

        ema_path = getattr(cfg.checkpoints, "ema_load_path", None)
        if ema_path and os.path.exists(ema_path):
            self.load_from_ema(ema_path)
        else:
            self.load_from_checkpoint(cfg.checkpoints.checkpoint_path)
        self._setup_vis_config()

    def _fail_fast_if_empty_dataset(self):
        try:
            ds_len = len(self.test_dataset)
        except Exception as e:
            print(f"[Diag][Dataset] len() failed: {type(e).__name__}: {e}")
            return

        if ds_len > 0:
            return

        val_cfg = self.cfg.dataset.val
        data_dirs = list(val_cfg.get("data_dirs", []))
        interval = val_cfg.get("time_interval", self.cfg.dataset.get("time_interval", None))
        n_past = val_cfg.get("n_past", self.cfg.dataset.get("n_past", None))
        n_futures = val_cfg.get("n_futures", self.cfg.dataset.get("n_futures", None))

        hint = ""
        if any("interval_30" in str(d) for d in data_dirs) and int(interval) == 10:
            hint = (
                "Detected possible mismatch: data_dirs contains `interval_30` but "
                "`time_interval` is 10. Try setting `dataset.val.time_interval: 30`."
            )

        raise RuntimeError(
            "[Diag] Test dataset is empty (len=0), so dataloader cannot yield batches.\n"
            f"  data_dirs={data_dirs}\n"
            f"  time_interval={interval}, n_past={n_past}, n_futures={n_futures}\n"
            "  Common causes: wrong time_interval for this data, or window length "
            "(n_past+n_futures) too large for consecutive sequences.\n"
            f"  {hint}"
        )

    def _run_diagnostics_before_accelerate(self):
        diag_cfg = self.cfg.test.get("diagnostics", {})
        enabled = bool(diag_cfg.get("enabled", True))
        if not enabled:
            return

        sample_count = int(diag_cfg.get("sample_count", 16))
        loader_batches = int(diag_cfg.get("loader_batches", 3))
        print(
            "[Diag] enabled: "
            f"sample_count={sample_count}, loader_batches={loader_batches}"
        )
        self._diagnose_dataset_samples(sample_count)
        self._diagnose_raw_loader(loader_batches)

    def _diagnose_dataset_samples(self, sample_count: int):
        total = len(self.test_dataset)
        if total <= 0:
            print("[Diag][Dataset] len=0 (no samples).")
            return

        n = max(1, min(sample_count, total))
        if n == total:
            indices = list(range(total))
        else:
            indices = sorted(set(int(round(i * (total - 1) / (n - 1))) for i in range(n)))

        ok, fail = 0, 0
        print(f"[Diag][Dataset] probing {len(indices)} / {total} samples...")
        for idx in indices:
            try:
                item = self.test_dataset[idx]
                if item is None:
                    fail += 1
                    print(f"[Diag][Dataset][FAIL] idx={idx} returned None")
                    continue
                if isinstance(item, dict):
                    missing = [k for k in ("radar_past", "satellite_past", "rain_past", "rain_future", "rain_future_cls") if k not in item]
                    if missing:
                        fail += 1
                        print(f"[Diag][Dataset][FAIL] idx={idx} missing keys: {missing}")
                        continue
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[Diag][Dataset][EXC] idx={idx} {type(e).__name__}: {e}")
        print(f"[Diag][Dataset] result: ok={ok}, fail={fail}")

    def _diagnose_raw_loader(self, max_batches: int):
        if max_batches <= 0:
            return
        print(f"[Diag][Loader] probing first {max_batches} batches before accelerate...")
        got = 0
        try:
            for bi, batch in enumerate(self.raw_test_loader):
                got += 1
                if batch is None:
                    print(f"[Diag][Loader][FAIL] batch#{bi} is None")
                    break
                if isinstance(batch, dict):
                    keys = sorted(list(batch.keys()))
                    bsz = None
                    if "rain_future" in batch and hasattr(batch["rain_future"], "shape"):
                        bsz = int(batch["rain_future"].shape[0])
                    print(f"[Diag][Loader] batch#{bi} keys={keys} bsz={bsz}")
                else:
                    print(f"[Diag][Loader] batch#{bi} type={type(batch)}")
                if got >= max_batches:
                    break
            if got == 0:
                print("[Diag][Loader][FAIL] no batch produced.")
        except Exception as e:
            print(f"[Diag][Loader][EXC] {type(e).__name__}: {e}")
            tb = traceback.format_exc(limit=2)
            print(tb)

    def load_from_checkpoint(self, ckpt_path):
        from accelerate import load_checkpoint_in_model

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if ckpt_path.is_dir():
            print(f"Loading accelerate checkpoint directory: {ckpt_path}")
            load_checkpoint_in_model(self.model, ckpt_path)
        else:
            print(f"Loading standard pytorch checkpoint file: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
        print(f"Successfully loaded checkpoint from: {ckpt_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True):
        ema_path = Path(ema_path)
        accelerate.load_checkpoint_in_model(self.model, ema_path / "rain_model", strict=strict)
        print(f"Loaded EMA weights from: {ema_path}")

    def _setup_vis_config(self):
        vis_cfg = self.cfg.test.get("vis", {})
        self.vis_enabled = bool(vis_cfg.get("enabled", False))
        self.vis_max_samples = int(vis_cfg.get("max_samples", 0))
        self.vis_n_past = int(vis_cfg.get("n_past_frames", int(self.cfg.dataset.get("n_past", 0))))
        self.vis_n_future = int(
            vis_cfg.get("n_future_frames", int(self.cfg.dataset.get("n_futures", 0)))
        )
        self.vis_dpi = int(vis_cfg.get("dpi", 180))
        self.vis_out_dir = Path(vis_cfg.get("out_dir", "outputs/test_vis_swinnet_cls"))
        self.vis_saved_count = 0
        self.vis_prefix = str(vis_cfg.get("file_prefix", "sample"))
        self.vis_only_strong = bool(vis_cfg.get("only_strong_samples", True))

        if self.vis_enabled and self.vis_max_samples > 0:
            self.vis_out_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[Vis] enabled, "
                f"max_samples={self.vis_max_samples}, "
                f"n_past_frames={self.vis_n_past}, "
                f"n_future_frames={self.vis_n_future}, "
                f"out_dir={self.vis_out_dir}"
            )

    @staticmethod
    def _to_class_map(x: torch.Tensor, num_classes: int, bounds: list[float]) -> torch.Tensor:
        """
        Convert [1,1,T,H,W] or [1,T,H,W] to class index map [T,H,W].
        If x is integer-like in [0, num_classes-1], keep as class ids.
        Otherwise bucketize by bounds.
        """
        if x.ndim == 5:
            x = x.squeeze(0).squeeze(0)
        elif x.ndim == 4:
            x = x.squeeze(0)
        x = x.detach().cpu()

        if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return x.long().clamp(0, num_classes - 1)

        x_rounded = x.round()
        is_class_like = torch.allclose(x, x_rounded, atol=1e-6) and x.min() >= 0 and x.max() <= (num_classes - 1)
        if is_class_like:
            return x_rounded.long().clamp(0, num_classes - 1)

        bounds_tensor = torch.tensor(bounds, dtype=x.dtype)
        cls = torch.bucketize(x.contiguous(), bounds_tensor, right=False) - 1
        return cls.clamp(0, num_classes - 1).long()

    def _save_visual_sample(
        self,
        rain_past_sample: torch.Tensor,
        rain_future_cls_sample: torch.Tensor,
        pred_cls_sample: torch.Tensor,
        bounds: list[float],
        batch_idx: int,
        sample_in_batch: int,
    ):
        if not self.vis_enabled or self.vis_saved_count >= self.vis_max_samples:
            return

        num_classes = len(bounds) - 1
        past_cls = self._to_class_map(rain_past_sample, num_classes, bounds)
        gt_cls = self._to_class_map(rain_future_cls_sample, num_classes, bounds)
        pred_cls = self._to_class_map(pred_cls_sample, num_classes, bounds)

        n_past = max(0, min(self.vis_n_past, past_cls.shape[0]))
        n_future = max(0, min(self.vis_n_future, gt_cls.shape[0], pred_cls.shape[0]))
        if n_past + n_future * 2 <= 0:
            return
        past_vis = past_cls[-n_past:] if n_past > 0 else past_cls[:0]

        # Keep exactly the same class-color mapping as trainer visualization.
        cmap_colors = [
            "white",
            "lightblue",
            "blue",
            "green",
            "yellow",
            "orange",
            "red",
            "purple",
            "black",
        ][:num_classes]
        cmap = mcolors.ListedColormap(cmap_colors)
        norm = mcolors.BoundaryNorm(np.arange(num_classes + 1), cmap.N)

        n_cols = n_past + n_future
        fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 6.2))
        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]], dtype=object)

        # First n_past columns: keep past frames in both rows for aligned comparison.
        for t in range(n_past):
            frame = past_vis[t].numpy()
            axes[0, t].imshow(frame, cmap=cmap, norm=norm)
            axes[0, t].set_title(f"Past t-{n_past - t}")
            axes[0, t].axis("off")

            axes[1, t].imshow(frame, cmap=cmap, norm=norm)
            axes[1, t].set_title(f"Past t-{n_past - t}")
            axes[1, t].axis("off")

        # Future columns: row-1 is GT, row-2 is Pred.
        for t in range(n_future):
            c = n_past + t
            axes[0, c].imshow(gt_cls[t].numpy(), cmap=cmap, norm=norm)
            axes[0, c].set_title(f"GT t+{t + 1}")
            axes[0, c].axis("off")

            axes[1, c].imshow(pred_cls[t].numpy(), cmap=cmap, norm=norm)
            axes[1, c].set_title(f"Pred t+{t + 1}")
            axes[1, c].axis("off")

        fig.text(0.01, 0.75, " GT", fontsize=11, rotation=90, va="center")
        fig.text(0.01, 0.25, " Pred", fontsize=11, rotation=90, va="center")
        fig.suptitle("Rain Class Visualization (Aligned GT vs Pred)", fontsize=12)
        plt.tight_layout()
        save_path = self.vis_out_dir / f"{self.vis_prefix}_b{batch_idx:04d}_i{sample_in_batch:02d}.png"
        plt.savefig(save_path, dpi=self.vis_dpi, bbox_inches="tight")
        plt.close(fig)
        self.vis_saved_count += 1
        print(f"[Vis] saved {save_path}")

    @torch.no_grad()
    def run(self):
        real_bounds = self.cfg.test.get("bounds", [0, 0.01, 0.1, 0.2, 0.5, 10])
        print(f"Using bounds strictly from YAML: {real_bounds}")

        num_classes = len(real_bounds) - 1
        is_cumulative = self.cfg.test.get("cumulative", True)
        tol_px = self.cfg.get("val", {}).get("tolerance_px", 0)
        cumulative_thresholds = [-1.0] + [i + 0.5 for i in range(num_classes - 1)]

        def make_metric_acc():
            if is_cumulative:
                return RainGlobalMetricsAccumulator(
                    bounds=cumulative_thresholds,
                    device=self.device,
                    tolerance_px=tol_px,
                )
            return StrictClassMetricsAccumulator(num_classes, self.device)

        mode_name = (
            f"Cumulative metrics (>=) | tolerance={tol_px}px"
            if is_cumulative
            else "Strict metrics (==)"
        )
        print(f"Metric mode: {mode_name}")

        strong_threshold = self.cfg.test.get("strong_threshold", 0)
        max_test_batches = int(self.cfg.test.get("max_test_batches", 0))
        if max_test_batches > 0:
            print(f"[Eval] Using max_test_batches={max_test_batches} (quick evaluation mode)")
        n_total, n_strong = 0, 0
        per_frame_metric_accs = None

        pbar = tqdm(self.test_dataloader, desc="Running global classification test")

        try:
            for bi, batch in enumerate(pbar):
                if max_test_batches > 0 and bi >= max_test_batches:
                    break
                radar_past = batch["radar_past"].to(self.device)
                sat_past = batch["satellite_past"].to(self.device)
                rain_past = batch["rain_past"].to(self.device)
                rain_future = batch["rain_future"].to(self.device)
                rain_future_cls = batch["rain_future_cls"].to(self.device)

                n_total += rain_future.size(0)

                has_strong = (rain_future >= strong_threshold).flatten(1).any(dim=1)
                idx = has_strong.nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue

                radar_past = radar_past[idx]
                sat_past = sat_past[idx]
                rain_past = rain_past[idx]
                rain_future_cls = rain_future_cls[idx]

                pred = self.model(radar_past, sat_past, rain_past)
                pred = pred.unsqueeze(2) if pred.ndim == 4 else pred
                pred_cls_idx = torch.argmax(pred, dim=1, keepdim=True).float()  # [B,1,T,H,W]
                gt_cls_idx = rain_future_cls.float()  # [B,1,T,H,W]

                t_future = pred_cls_idx.shape[2]
                if per_frame_metric_accs is None:
                    per_frame_metric_accs = [make_metric_acc() for _ in range(t_future)]

                for t in range(t_future):
                    pred_t = pred_cls_idx[:, :, t : t + 1]
                    gt_t = gt_cls_idx[:, :, t : t + 1]
                    per_frame_metric_accs[t].update(pred_t, gt_t)

                if (
                    self.vis_enabled
                    and self.vis_max_samples > 0
                    and self.vis_saved_count < self.vis_max_samples
                    and self.accelerator.is_main_process
                ):
                    for local_i in range(pred_cls_idx.shape[0]):
                        if self.vis_saved_count >= self.vis_max_samples:
                            break
                        if (not self.vis_only_strong) or has_strong[idx][local_i].item():
                            self._save_visual_sample(
                                rain_past_sample=rain_past[local_i : local_i + 1],
                                rain_future_cls_sample=rain_future_cls[local_i : local_i + 1],
                                pred_cls_sample=pred_cls_idx[local_i : local_i + 1],
                                bounds=real_bounds,
                                batch_idx=bi,
                                sample_in_batch=local_i,
                            )

                n_strong += idx.numel()
                pbar.set_postfix({"strong_samples": n_strong, "T_future": t_future})
        except ValueError as e:
            if "Batch does not contain any data" in str(e):
                print("[Diag][Accelerate] Caught empty-batch ValueError during iteration.")
                print("[Diag][Hint] Try reducing `dataset.val.num_workers` to 0 or 2 to verify worker-sharding/reader stability.")
                print("[Diag][Hint] If raw loader probe passed but prepared loader failed, issue is likely Accelerate+iterable-dataset interaction.")
            raise

        if n_strong == 0:
            print("\nNo strong rain samples found in test set!")
            return

        if per_frame_metric_accs is None:
            print("\nNo valid predictions for metric computation.")
            return

        title = "Cumulative (>=)" if is_cumulative else "Strict (==)"
        print(f"\n===== Per-frame Metrics: {title} =====")
        print(f"Strong-rain samples: {n_strong} / {n_total}")
        if self.vis_enabled:
            print(f"Visualization saved samples: {self.vis_saved_count}")

        for t, metric_acc in enumerate(per_frame_metric_accs):
            metrics = metric_acc.compute()
            print(f"\n--- Future Frame t+{t + 1} ---")
            for th_key, vals in metrics.items():
                if is_cumulative:
                    try:
                        th_val_internal = float(th_key.replace(">=", "").replace("mm", ""))
                        cls_id = int(th_val_internal + 0.5)
                        if cls_id < len(real_bounds):
                            readable_key = f"Rain >= {real_bounds[cls_id]} (Class {cls_id})"
                        else:
                            readable_key = f"Class >= {cls_id}"
                    except Exception:
                        readable_key = th_key
                else:
                    cls_id = int(th_key.split("_")[-1])
                    if cls_id < len(real_bounds) - 1:
                        lower = real_bounds[cls_id]
                        upper = real_bounds[cls_id + 1]
                        readable_key = f"Rain in [{lower}, {upper}) (Strict Class {cls_id})"
                    else:
                        readable_key = f"Class == {cls_id}"

                msg = " ".join([f"{k}={v.item():.4f}" for k, v in vals.items()])
                print(f"{readable_key}: {msg}")


@hydra.main(config_path="../config/ts_rain_test", config_name="rain_test_ts_swinnet_cls", version_base=None)
def main(cfg):
    tester = RainPredictionTester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
