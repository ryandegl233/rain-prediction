import copy
import math

import torch


class HorizonController:
    def __init__(
        self,
        enabled: bool = True,
        warmup_steps: int = 1000,
        update_every: int = 200,
        ema_decay: float = 0.8,
        alpha: float = 0.5,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        if update_every <= 0 or warmup_steps < update_every:
            raise ValueError("warmup_steps must cover at least one positive update window.")
        if not 0 <= ema_decay < 1 or not 0 <= alpha <= 0.5:
            raise ValueError("Invalid EMA decay or adaptive weight budget.")
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("temperature and eps must be finite and positive.")
        self.settings = dict(
            enabled=enabled,
            warmup_steps=warmup_steps,
            update_every=update_every,
            ema_decay=ema_decay,
            alpha=alpha,
            temperature=temperature,
            eps=eps,
        )
        self.step = 0
        self.weights = torch.full((4,), 0.25)
        self.reference = torch.zeros(4)
        self.ema = torch.zeros(4)
        self.window_sum = torch.zeros(4)
        self.window_count = 0
        self.neutral_reference_count = 0

    def observe(self, error_sum: torch.Tensor, count: int, step: int) -> None:
        value = error_sum.detach().to(device="cpu", dtype=torch.float32)
        if value.shape != (4,) or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("Expected four finite nonnegative error sums.")
        if count <= 0 or step != self.step + 1:
            raise ValueError("Expected a positive count and the next successful optimizer step.")
        cfg = self.settings
        warmup, window = cfg["warmup_steps"], cfg["update_every"]
        self.step = step
        if step <= warmup - window:
            return
        self.window_sum += value
        self.window_count += count
        if step < warmup or (step - warmup) % window:
            return
        mean = self.window_sum / self.window_count
        if not torch.isfinite(mean).all():
            raise ValueError("Nonfinite accumulated controller statistics.")
        if step == warmup:
            self.reference = mean.clone()
            self.ema = mean.clone()
            self.neutral_reference_count = int((mean < cfg["eps"]).sum())
        else:
            self.ema = cfg["ema_decay"] * self.ema + (1 - cfg["ema_decay"]) * mean
            ratio = (self.ema / self.reference.clamp_min(cfg["eps"])).clamp(0.25, 4.0)
            ratio = torch.where(self.reference < cfg["eps"], torch.ones_like(ratio), ratio)
            if cfg["enabled"]:
                adaptive = torch.softmax(ratio.log() / cfg["temperature"], dim=0)
                self.weights = (1 - cfg["alpha"]) / 4 + cfg["alpha"] * adaptive
        self.window_sum.zero_()
        self.window_count = 0

    def state_dict(self) -> dict[str, object]:
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.keys() != vars(self).keys() or state["settings"] != self.settings:
            raise ValueError("Controller checkpoint/configuration mismatch.")
        for name in ("weights", "reference", "ema", "window_sum"):
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (4,)
                or not torch.isfinite(value).all()
                or (value < 0).any()
            ):
                raise ValueError(f"Invalid controller checkpoint tensor: {name}")
        if not torch.isclose(state["weights"].sum(), torch.tensor(1.0)):
            raise ValueError("Checkpoint weights do not sum to one.")
        if state["step"] < 0 or state["window_count"] < 0:
            raise ValueError("Invalid controller checkpoint counters.")
        self.__dict__.update(copy.deepcopy(state))
