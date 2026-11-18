import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.baselines_stage3.deep.earth_alpha import build_earth_alpha


ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "src/baselines_stage3/deep/earth_alpha_port"
UPSTREAM = ROOT.parent / "third_party/baselines"
TINY_EARTH = dict(image_size=16, base_units=16, stem_dims=[4, 8, 16], stem_scales=[2, 2, 2], dropout=0)


@pytest.mark.parametrize("method", ["earthformer", "alphapre"])
def test_forward_backward_optimizer(method: str) -> None:
    torch.set_num_threads(1)
    options = TINY_EARTH if method == "earthformer" else dict(image_size=16, hidden_dim=8)
    core = build_earth_alpha(method, options)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)
    history = torch.randn(2, 12, 7, 16, 16)
    prediction, auxiliary = core(history)
    assert prediction.shape == (2, 12, 1, 16, 16)
    loss = prediction.square().mean()
    if method == "alphapre":
        losses = core.supervised_losses(auxiliary, torch.rand_like(prediction), step=0)
        assert all(value.ndim == 0 for value in losses.values())
        loss = loss + sum(losses.values())
    loss.backward()
    gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    parameter = next(parameter for parameter in core.parameters() if parameter.grad is not None)
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter)


@pytest.mark.parametrize("method", ["earthformer", "alphapre"])
def test_parameter_budget(method: str) -> None:
    with torch.device("meta"):
        core = build_earth_alpha(method, {})
    parameters = sum(parameter.numel() for parameter in core.parameters())
    assert abs(parameters / 44_162_136 - 1) < 0.05


def test_vendor_source_checksums() -> None:
    records = json.loads((PORT / "SOURCE_MANIFEST.json").read_text())
    for record in records:
        vendor = PORT / record["path"]
        assert hashlib.sha256(vendor.read_bytes()).hexdigest() == record["sha256"]
        original = UPSTREAM / record["repository"] / record["source"]
        if original.exists():
            assert vendor.read_bytes() == original.read_bytes()


def test_alphapre_original_outputs_and_supervision() -> None:
    original = UPSTREAM / "AlphaPre/models/alphapre.py"
    if not original.exists():
        pytest.skip("Original checkout is not available; vendored checksums remain verified")
    specification = importlib.util.spec_from_file_location("alphapre_original_reference", original)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    core = build_earth_alpha("alphapre", dict(image_size=16, hidden_dim=8))
    reference = module.AlphaPre(**core.model_options)
    reference.load_state_dict(core.network.state_dict())
    history = torch.rand(2, 12, 7, 16, 16)
    target = torch.rand(2, 12, 1, 16, 16)
    prediction, auxiliary = core(history)
    expected, expected_losses = reference.predict(history.transpose(1, 2), target.transpose(1, 2), True)
    torch.testing.assert_close(prediction.transpose(1, 2), expected, rtol=0, atol=0)
    losses = core.supervised_losses(auxiliary, target, step=0)
    for actual, original_key in [
        ("phase_loss", "phase_loss"),
        ("amplitude_loss", "ampli_loss"),
        ("anet_loss", "anet_loss"),
    ]:
        torch.testing.assert_close(losses[actual], expected_losses[original_key])
    assert core.supervised_losses(auxiliary, target, step=9999)["amplitude_loss"].item() == 0


def test_earthformer_original_outputs(tmp_path: Path) -> None:
    original = UPSTREAM / "earth-forecasting-transformer/src"
    if not original.exists():
        pytest.skip("Original checkout is not available; vendored checksums remain verified")
    core = build_earth_alpha("earthformer", TINY_EARTH).eval()
    history = torch.rand(2, 12, 7, 16, 16)
    payload = tmp_path / "reference_input.pt"
    output = tmp_path / "reference_output.pt"
    torch.save(dict(options=core.model_options, state=core.network.state_dict(), history=history), payload)
    script = (
        "import sys,torch; "
        "from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel; "
        "torch.set_num_threads(1); "
        "p=torch.load(sys.argv[1],weights_only=False); "
        "m=CuboidTransformerModel(**p['options']).eval(); m.load_state_dict(p['state']); "
        "torch.save(m(p['history'].permute(0,2,3,4,1).contiguous()).detach(),sys.argv[2])"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(original) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script, str(payload), str(output)], env=environment, check=True)
    prediction, _ = core(history)
    expected = torch.load(output, weights_only=True).permute(0, 4, 1, 2, 3).contiguous()
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize("size", [28, 448])
def test_alphapre_bfloat16_autocast_non_power_of_two(size: int) -> None:
    torch.set_num_threads(1)
    core = build_earth_alpha("alphapre", dict(image_size=size, hidden_dim=8)).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output, auxiliary = core(torch.randn(1, 12, 7, size, size, dtype=torch.bfloat16))
    assert output.shape == (1, 12, 1, size, size)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    assert torch.isfinite(auxiliary["phase_prediction"]).all()
