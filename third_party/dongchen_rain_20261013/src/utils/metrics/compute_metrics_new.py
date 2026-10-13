"""
Rainfall Level Definitions and Thresholds

This module defines rainfall intensity thresholds used for classification metrics.
Different threshold configurations are provided for various time resolutions.

Threshold Configurations:
    - BOUNDS = [0, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 7.5]  # mm/hour
    - BOUNDS = np.array([0, 0.0001, 0.007, 0.017, 0.035, 0.069, 0.17])  # mm/min
    - BOUNDS = np.array([0, 0.003, 0.21, 0.51, 1.05, 2.07, 5.07])  # mm/30min

Rainfall Intensity Levels (China Meteorological Administration standards):
    - Light rain: 0.1-4.9mm in 12h or 0.1-9.9mm in 24h
    - Moderate rain: 5.0-14.9mm in 12h or 10-24.9mm in 24h
    - Heavy rain: 15.0-29.9mm in 12h or 25.0-49.9mm in 24h
    - Storm: 30.0-69.9mm in 12h or 50.0-99.9mm in 24h
    - Heavy storm: 70.0-139.9mm in 12h or 100.0-249.9mm in 24h
    - Severe storm: ≥140.0mm in 12h or ≥250.0mm in 24h

Author: Zihan Cao, Jieyi Zhu, Dongchen Wang
Institution: UESTC
"""

import time

import torch
import torch.nn.functional as F
try:
    import torchmetrics
except ModuleNotFoundError:  # optional dependency for RainPredMetrics only
    torchmetrics = None

__all__ = [
    "RainGlobalMetricsAccumulator",
    "RainPredMetrics",
]


BOUNDS = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class RainGlobalMetricsAccumulator:
    """
    Global accumulator for rainfall threshold classification metrics.

    Description:
    - Performs binary classification for each pixel at multiple rainfall thresholds,
      accumulating TP/FP/FN/TN across all data.
    - Supports spatial tolerance: allows predictions to match within GT neighborhood
      (useful for sparse moderate/heavy rain pixels).
    - Suitable for large-scale offline statistics, computing global metrics at once.

    Args:
        bounds (list[float]): List of thresholds, e.g., [0, 0.01, 0.1, 0.2, ...].
            Skips the first threshold, using bounds[1:] as actual classification thresholds.
            Units should match pred/target (e.g., mm/30min or mm/h).
        device (torch.device | str | None): Device for statistic tensors.
            Defaults to CPU if None.
        tolerance_px (int): Spatial tolerance radius in pixels.
            0 means strict pixel-wise matching (default).
            For example, tolerance_px=2 uses a (2*2+1)=5 window for neighborhood tolerance.
        tolerance_min_th (float | None): Enable spatial tolerance only for thresholds >= this value.
            None means tolerance is enabled for all thresholds.
            For example, setting to 0.3 enables tolerance only for moderate/heavy rain,
            keeping light rain matching strict.

    Returns:
        Returns CSI/POD/FAR/F1/HSS metrics for each threshold via compute().
    """

    def __init__(self, bounds=BOUNDS, device=None, tolerance_px=0, tolerance_min_th=None):
        self.bounds = bounds
        self.device = device or torch.device("cpu")
        self.tolerance_px = int(tolerance_px)
        self.tolerance_min_th = tolerance_min_th
        self.stats = {
            f">={th}mm": {
                "TP": torch.tensor(0.0, device=self.device),
                "FP": torch.tensor(0.0, device=self.device),
                "FN": torch.tensor(0.0, device=self.device),
                "TN": torch.tensor(0.0, device=self.device),
                "N_true": torch.tensor(0.0, device=self.device),
            }
            for th in self.bounds
        }

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: Predicted tensor with shape [B, 1, T, H, W]
            target: Ground truth tensor with shape [B, 1, T, H, W]
        """
        assert pred.ndim == 5 and target.ndim == 5, "expect (B,C,T,H,W)"
        B, C, T, H, W = pred.shape

        use_tolerance = self.tolerance_px > 0
        if not use_tolerance:
            pred_flat = pred.reshape(-1)
            target_flat = target.reshape(-1)

            for th in self.bounds[1:]:
                pred_bin = (pred_flat >= th).int()
                target_bin = (target_flat >= th).int()

                TP = (pred_bin * target_bin).sum().float()
                FP = (pred_bin * (1 - target_bin)).sum().float()
                FN = ((1 - pred_bin) * target_bin).sum().float()
                TN = ((1 - pred_bin) * (1 - target_bin)).sum().float()
                N_true = target_bin.sum().float()

                s = self.stats[f">={th}mm"]
                s["TP"] += TP
                s["FP"] += FP
                s["FN"] += FN
                s["TN"] += TN
                s["N_true"] += N_true
            return

        k = 2 * self.tolerance_px + 1
        for th in self.bounds[1:]:
            use_tol_for_th = self.tolerance_min_th is None or th >= self.tolerance_min_th

            pred_bin = pred >= th
            target_bin = target >= th

            if not use_tol_for_th:
                pred_flat = pred_bin.reshape(-1).int()
                target_flat = target_bin.reshape(-1).int()

                TP = (pred_flat * target_flat).sum().float()
                FP = (pred_flat * (1 - target_flat)).sum().float()
                FN = ((1 - pred_flat) * target_flat).sum().float()
                TN = ((1 - pred_flat) * (1 - target_flat)).sum().float()
                N_true = target_flat.sum().float()
            else:
                pred_2d = pred_bin.reshape(B * T, 1, H, W)
                target_2d = target_bin.reshape(B * T, 1, H, W)

                pred_dil = F.max_pool2d(pred_2d.float(), k, stride=1, padding=self.tolerance_px) > 0
                target_dil = F.max_pool2d(target_2d.float(), k, stride=1, padding=self.tolerance_px) > 0

                TP = (pred_2d & target_dil).sum().float()
                FP = (pred_2d & ~target_dil).sum().float()
                FN = (target_2d & ~pred_dil).sum().float()
                TN = (~pred_2d & ~target_2d).sum().float()
                N_true = target_2d.sum().float()

            s = self.stats[f">={th}mm"]
            s["TP"] += TP
            s["FP"] += FP
            s["FN"] += FN
            s["TN"] += TN
            s["N_true"] += N_true

    @torch.no_grad()
    def compute(self):
        """
        Call after accumulating all batches to compute global metrics at once.
        """
        results = {}
        print("\n===== Debug: Valid Pixel Distribution =====")
        for th, s in self.stats.items():
            TP, FP, FN, TN, N_true = (s["TP"], s["FP"], s["FN"], s["TN"], s["N_true"])

            print(f"[{th}] Valid target pixels: {int(N_true.item())}")

            if (TP + FN) == 0:
                print(f" Skip {th} (no positive samples)")
                continue

            csi = TP / (TP + FP + FN + 1e-8)
            pod = TP / (TP + FN + 1e-8)
            far = FP / (TP + FP + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            hss = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + 1e-8)

            results[th] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "F1": f1,
                "HSS": hss,
            }

        if not results:
            print("\n No thresholds have positive samples, all metrics are empty.")
        return results


class RainPredMetrics(torchmetrics.Metric if torchmetrics is not None else object):
    """
    TorchMetrics-style rainfall threshold classification metrics calculator.

    Description:
    - Accumulates TP/FP/FN/TN independently for each threshold.
    - Supports spatial tolerance for more reasonable matching of sparse moderate/heavy rain pixels.
    - Can be called directly in training/validation loops via update/compute.

    Args:
        bounds (list[float]): List of thresholds, using bounds[1:] as actual classification thresholds.
            Units should match pred/target.
        tolerance_px (int): Spatial tolerance radius in pixels.
            0 means strict pixel-wise matching (default).
            >0 means predictions falling within GT neighborhood are also counted as TP.
        tolerance_min_th (float | None): Enable spatial tolerance only for thresholds >= this value.
            None means tolerance is enabled for all thresholds.

    Example:
        metric = RainPredMetrics(bounds=BOUNDS, tolerance_px=2, tolerance_min_th=0.3)
        metric.update(pred, target)
        results = metric.compute()
    """

    def __init__(self, bounds=BOUNDS, tolerance_px=0, tolerance_min_th=None):
        if torchmetrics is None:
            raise ModuleNotFoundError("torchmetrics is required for RainPredMetrics")
        super().__init__()
        self.register_buffer("thresholds", torch.tensor(bounds[1:], dtype=torch.float32))
        self.tolerance_px = int(tolerance_px)
        self.tolerance_min_th = tolerance_min_th
        num_th = self.thresholds.numel()
        self.add_state("tp", default=torch.zeros(num_th), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_th), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_th), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_th), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: Predicted tensor with shape [B, 1, T, H, W]
            target: Ground truth tensor with shape [B, 1, T, H, W]
        """
        assert pred.ndim == 5 and target.ndim == 5, "expect (B,C,T,H,W)"
        B, C, T, H, W = pred.shape
        thresholds = self.thresholds.to(pred.device)

        if self.tolerance_px <= 0:
            pred_flat = pred.reshape(-1)
            target_flat = target.reshape(-1)

            pred_bin = pred_flat[None, :] >= thresholds[:, None]
            target_bin = target_flat[None, :] >= thresholds[:, None]

            tp = (pred_bin & target_bin).sum(dim=1, dtype=torch.float32)
            fp = (pred_bin & ~target_bin).sum(dim=1, dtype=torch.float32)
            fn = (~pred_bin & target_bin).sum(dim=1, dtype=torch.float32)
            tn = (~pred_bin & ~target_bin).sum(dim=1, dtype=torch.float32)

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
            return

        k = 2 * self.tolerance_px + 1
        for i, th in enumerate(thresholds):
            use_tol_for_th = self.tolerance_min_th is None or float(th) >= self.tolerance_min_th

            pred_bin = pred >= th
            target_bin = target >= th

            if not use_tol_for_th:
                pred_flat = pred_bin.reshape(-1)
                target_flat = target_bin.reshape(-1)

                self.tp[i] += (pred_flat & target_flat).sum(dtype=torch.float32)
                self.fp[i] += (pred_flat & ~target_flat).sum(dtype=torch.float32)
                self.fn[i] += (~pred_flat & target_flat).sum(dtype=torch.float32)
                self.tn[i] += (~pred_flat & ~target_flat).sum(dtype=torch.float32)
                continue

            pred_2d = pred_bin.reshape(B * T, 1, H, W)
            target_2d = target_bin.reshape(B * T, 1, H, W)

            pred_dil = F.max_pool2d(pred_2d.float(), k, stride=1, padding=self.tolerance_px) > 0
            target_dil = F.max_pool2d(target_2d.float(), k, stride=1, padding=self.tolerance_px) > 0

            self.tp[i] += (pred_2d & target_dil).sum(dtype=torch.float32)
            self.fp[i] += (pred_2d & ~target_dil).sum(dtype=torch.float32)
            self.fn[i] += (target_2d & ~pred_dil).sum(dtype=torch.float32)
            self.tn[i] += (~pred_2d & ~target_2d).sum(dtype=torch.float32)

    @torch.no_grad()
    def compute(self):
        results = {}
        for i, th in enumerate(self.thresholds):
            tp = self.tp[i]
            fp = self.fp[i]
            fn = self.fn[i]
            tn = self.tn[i]

            if (tp + fn) == 0:
                continue

            csi = tp / (tp + fp + fn + 1e-8)
            pod = tp / (tp + fn + 1e-8)
            far = fp / (tp + fp + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-8)

            results[f">={th.item()}mm"] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "F1": f1,
                "HSS": hss,
            }

        return results


# ------------ Test cases ------------- #


def test_diff_implem_equlty():
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 3, 8, 8)
    target = torch.rand(2, 1, 3, 8, 8)

    acc = RainGlobalMetricsAccumulator(bounds=BOUNDS, device="cpu")
    acc.update(pred, target)
    global_results = acc.compute()

    metric = RainPredMetrics(bounds=BOUNDS)
    metric.update(pred, target)
    tm_results = metric.compute()

    def _normalize(results):
        normalized = {}
        for key, vals in results.items():
            th = float(key[2:-2])
            normalized[round(th, 4)] = vals
        return normalized

    global_norm = _normalize(global_results)
    tm_norm = _normalize(tm_results)
    max_err = -1.0
    max_err_key = None

    for th in BOUNDS[1:]:
        key = round(float(th), 4)
        if key not in global_norm or key not in tm_norm:
            continue
        for name in ["CSI", "POD", "FAR", "F1", "HSS"]:
            diff = (global_norm[key][name] - tm_norm[key][name]).abs().max().item()
            if diff > max_err:
                max_err = diff
                max_err_key = (key, name)

            t1 = global_norm[key][name].cpu()
            t2 = tm_norm[key][name].cpu()
            print(f"name: {name}, th: {key}, t1: {t1}, t2: {t2}")

            torch.testing.assert_close(
                t1,
                t2,
                rtol=1e-5,
                atol=1e-6,
            )

    if max_err_key is not None:
        th, name = max_err_key
        gv = global_norm[th][name].item()
        tv = tm_norm[th][name].item()
        print(f"max_err={max_err:.6g} at th={th} metric={name}")
        print(f"values: global={gv:.6g} torchmetrics={tv:.6g}")
    else:
        print("no comparable thresholds; max_err not computed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    large_pred = torch.rand(1, 1, 1, 512, 512, device=device)
    large_target = torch.rand(1, 1, 1, 512, 512, device=device)

    t0 = time.perf_counter()
    acc_large = RainGlobalMetricsAccumulator(bounds=BOUNDS, device=device)
    acc_large.update(large_pred, large_target)
    acc_large.compute()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    metric_large = RainPredMetrics(bounds=BOUNDS).to(device)
    metric_large.update(large_pred, large_target)
    metric_large.compute()
    t3 = time.perf_counter()

    print(f"invoke_time_global={t1 - t0:.6f}s torchmetrics={t3 - t2:.6f}s")


def test_diff_with_tolerance_pxs():
    bounds = [0, 0.5]
    pred = torch.zeros(1, 1, 1, 5, 5)
    target = torch.zeros_like(pred)
    target[0, 0, 0, 2, 2] = 1.0
    pred[0, 0, 0, 2, 3] = 1.0

    acc_strict = RainGlobalMetricsAccumulator(bounds=bounds, device="cpu", tolerance_px=0)
    acc_strict.update(pred, target)
    global_strict = acc_strict.compute()

    metric_strict = RainPredMetrics(bounds=bounds, tolerance_px=0)
    metric_strict.update(pred, target)
    tm_strict = metric_strict.compute()

    strict_global = global_strict[">=0.5mm"]
    strict_tm = tm_strict[">=0.5mm"]
    for name in ["CSI", "POD", "FAR", "F1", "HSS"]:
        torch.testing.assert_close(
            strict_global[name].cpu(),
            strict_tm[name].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )

    torch.testing.assert_close(strict_global["CSI"].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(strict_global["POD"].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(strict_global["FAR"].cpu(), torch.tensor(1.0))
    torch.testing.assert_close(strict_global["F1"].cpu(), torch.tensor(0.0))

    acc_tol = RainGlobalMetricsAccumulator(bounds=bounds, device="cpu", tolerance_px=1)
    acc_tol.update(pred, target)
    global_tol = acc_tol.compute()

    metric_tol = RainPredMetrics(bounds=bounds, tolerance_px=1)
    metric_tol.update(pred, target)
    tm_tol = metric_tol.compute()

    tol_global = global_tol[">=0.5mm"]
    tol_tm = tm_tol[">=0.5mm"]
    for name in ["CSI", "POD", "FAR", "F1", "HSS"]:
        torch.testing.assert_close(
            tol_global[name].cpu(),
            tol_tm[name].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )

    torch.testing.assert_close(tol_global["CSI"].cpu(), torch.tensor(1.0))
    torch.testing.assert_close(tol_global["POD"].cpu(), torch.tensor(1.0))
    torch.testing.assert_close(tol_global["FAR"].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(tol_global["F1"].cpu(), torch.tensor(1.0))
