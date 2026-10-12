# Band-limited occurrence gate

This experiment preserves the frozen deterministic predictor and trains only a pixel occurrence head.

For an ungated prediction `r` and occurrence probability `p`, only `0.01 <= r < 0.1 mm/min` is adjusted:

```text
w(r) = clamp((0.1 - r) / 0.09, 0, 1)
r_out = r * (1 - lambda * w(r) * (1 - p)^gamma)
```

Values below `0.01` and at or above `0.1 mm/min` are exact identity paths. Autoregressive history always receives `r`, not `r_out`, so occurrence postprocessing cannot accumulate through t+1 to t+4.

The default matched experiment trains one occurrence head for 1200 optimizer steps with learning rate `1e-4`, physical batch size 2, and gradient accumulation 4 (effective batch 8). It then evaluates the same weights at maximum suppression strengths `lambda = 0.25, 0.50, 0.75`. Reports assert exact invariance of CSI/POD/FAR/F1/HSS at thresholds `0.1`, `0.2`, and `0.5` before accepting a result.
