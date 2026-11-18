import torch

class GlobalMetricsAccumulator:
    def __init__(self, num_classes: int, device=None):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")

        # 每个类别独立统计
        self.stats = {
            cls_id: {
                "TP": torch.tensor(0.0, device=self.device),
                "FP": torch.tensor(0.0, device=self.device),
                "FN": torch.tensor(0.0, device=self.device),
                "TN": torch.tensor(0.0, device=self.device),
            }
            for cls_id in range(num_classes)
        }

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        累加一次 batch 的混淆统计。
        pred: [B, num_classes, T, H, W]
        target: [B, T, H, W] 或 [B, 1, T, H, W]
        """
        assert pred.ndim == 5, "expect (B,C,T,H,W)"
        if target.ndim == 4:
            target = target.unsqueeze(1)

        B, C, T, H, W = pred.shape
        device = pred.device
        pred_cls = torch.argmax(pred, dim=1, keepdim=True)  # [B,1,T,H,W]

        pred_cls_flat = pred_cls.view(-1, 1, H, W)
        target_flat = target.view(-1, 1, H, W)

        # 每类统计
        for cls_id in range(self.num_classes):
            p = (pred_cls_flat == cls_id).int()
            g = (target_flat == cls_id).int()
            self.stats[cls_id]["TP"] += (p * g).sum().float()
            self.stats[cls_id]["FP"] += (p * (1 - g)).sum().float()
            self.stats[cls_id]["FN"] += ((1 - p) * g).sum().float()
            self.stats[cls_id]["TN"] += ((1 - p) * (1 - g)).sum().float()

    @torch.no_grad()
    def compute(self):
        """
        累计完所有 batch 后调用，一次性计算全局指标。
        """
        results = {}
        for cls_id, s in self.stats.items():
            TP, FP, FN, TN = s["TP"], s["FP"], s["FN"], s["TN"]

            # 若该类在整个测试集都不存在
            if (TP + FP + FN) == 0:
                results[f"class_{cls_id}"] = {
                    "CSI": torch.tensor(0.0, device=self.device),
                    "POD": torch.tensor(0.0, device=self.device),
                    "FAR": torch.tensor(1.0, device=self.device),
                    "F1":  torch.tensor(0.0, device=self.device),
                    "HSS": torch.tensor(0.0, device=self.device),
                }
                continue

            csi = TP / (TP + FP + FN + 1e-8)
            pod = TP / (TP + FN + 1e-8)
            far = FP / (TP + FP + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            hss = 2 * (TP * TN - FP * FN) / (
                (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + 1e-8
            )

            results[f"class_{cls_id}"] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "F1": f1,
                "HSS": hss,
            }
        return results

if __name__ == "__main__":
    B, num_classes, T, H, W = 8, 6, 10, 256, 256
    acc = GlobalMetricsAccumulator(num_classes=num_classes, device="cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(5):
        pred = torch.randn(B, num_classes, T, H, W)
        target = torch.randint(0, num_classes, (B, T, H, W))
        acc.update(pred, target)

    metrics = acc.compute()
    print("\n===== Global 累积分类任务指标 =====")
    for k, v in metrics.items():
        print(f"\n类别 {k}:")
        for m_name, m_val in v.items():
            print(f"  {m_name:8s}: {m_val.item():.4f}")
