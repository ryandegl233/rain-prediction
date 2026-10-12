import math
from typing import Any

from torch import nn

from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame


class RainCausalPatchTransformerCCCInitFixed(RainCausalPatchTransformerNextFrame):
    def __init__(self, ccc_condition_scale_init: float = 0.1, **kwargs: Any) -> None:
        if not math.isfinite(ccc_condition_scale_init) or not 0 < ccc_condition_scale_init <= 1:
            raise ValueError("ccc_condition_scale_init must be finite and in (0, 1]")
        super().__init__(**kwargs)
        if self.cross_modal_adapter is not None:
            calibration = self.cross_modal_adapter.consensus_calibration
            if calibration is not None:
                nn.init.constant_(calibration.condition_scale, ccc_condition_scale_init)
