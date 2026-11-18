import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.baselines_stage3.deep import openstl
from src.baselines_stage3.deep.openstl import _patchify, _unpatchify, build_openstl
from src.baselines_stage3.deep.openstl_port.convlstm_cell import ConvLSTMCell
from src.baselines_stage3.deep.openstl_port.predrnnv2_cell import SpatioTemporalLSTMCellv2


PORT_ROOT = Path(openstl.__file__).parent / "openstl_port"
TINY_OPTIONS: dict[str, dict[str, object]] = {
    "convlstm": {"num_hidden": [8, 8], "patch_size": 2, "filter_size": 3},
    "predrnn_v2": {"num_hidden": [8, 8], "patch_size": 2, "filter_size": 3},
    "simvp": {"hid_S": 8, "hid_T": 16, "N_S": 2, "N_T": 2},
}


def official_definitions(filename: str, names: set[str], namespace: dict[str, object]) -> dict[str, object]:
    path = PORT_ROOT / "reference" / filename
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("method", TINY_OPTIONS)
@pytest.mark.parametrize("image_size", [16, 32])
def test_openstl_forward_backward_and_update(method: str, image_size: int) -> None:
    torch.manual_seed(3)
    model = build_openstl(method, TINY_OPTIONS[method] | {"image_size": image_size})
    history = torch.rand(2, 12, 7, image_size, image_size, requires_grad=True)
    prediction, auxiliary = model(history)
    assert prediction.shape == (2, 12, 1, image_size, image_size)
    assert torch.isfinite(prediction).all()
    assert set(auxiliary) == ({"decoupling"} if method == "predrnn_v2" else set())
    loss = prediction.square().mean() + sum(0.1 * value for value in auxiliary.values())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss.backward()
    assert history.grad is not None and torch.isfinite(history.grad).all() and history.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        unused = (
            method == "convlstm" and name.startswith("cell_list.") and (".conv_o." in name or ".conv_last." in name)
        )
        if unused:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    if method == "predrnn_v2":
        assert model.adapter.weight.grad.abs().sum() > 0
    updated_parameter = next(parameter for parameter in model.parameters() if parameter.grad is not None)
    before = updated_parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, updated_parameter)


@pytest.mark.parametrize("method", ["convlstm", "predrnn_v2"])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_recurrent_matches_official_prediction_and_decoupling(method: str, layer_norm: bool) -> None:
    torch.manual_seed(7)
    options = TINY_OPTIONS[method] | {"image_size": 16, "layer_norm": layer_norm}
    model = build_openstl(method, options)
    history = torch.rand(2, 12, 7, 16, 16)
    config = SimpleNamespace(
        in_shape=(7, 12, 16, 16),
        patch_size=2,
        filter_size=3,
        stride=1,
        layer_norm=layer_norm,
        pre_seq_length=7,
        aft_seq_length=1,
        total_length=8,
        reverse_scheduled_sampling=0,
        decouple_beta=0.1,
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "ConvLSTMCell": ConvLSTMCell,
        "SpatioTemporalLSTMCellv2": SpatioTemporalLSTMCellv2,
    }
    classname = "ConvLSTM_Model" if method == "convlstm" else "PredRNNv2_Model"
    filename = "convlstm_model.py" if method == "convlstm" else "predrnnv2_model.py"
    reference = official_definitions(filename, {classname}, namespace)[classname](2, [8, 8], config)
    reference.load_state_dict(model.state_dict(), strict=True)
    transform = official_definitions("predrnn_utils.py", {"reshape_patch", "reshape_patch_back"}, {})
    full_frames = torch.cat([history, torch.zeros(2, 12, 1, 16, 16)], dim=2).permute(0, 2, 3, 4, 1)
    patch_frames = transform["reshape_patch"](full_frames, 2)
    official_prediction, official_loss = reference(patch_frames, patch_frames[:, :0], return_loss=True)
    expected = transform["reshape_patch_back"](official_prediction[:, -1:], 2).permute(0, 4, 1, 2, 3)
    prediction, auxiliary = model(history)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    if method == "predrnn_v2":
        official_mse = F.mse_loss(official_prediction, patch_frames[:, 1:])
        expected_decoupling = (official_loss - official_mse) / config.decouple_beta
        torch.testing.assert_close(auxiliary["decoupling"], expected_decoupling, rtol=2e-6, atol=2e-6)
        reference.zero_grad()
        model.zero_grad()
        expected_decoupling.backward()
        auxiliary["decoupling"].backward()
        torch.testing.assert_close(model.adapter.weight.grad, reference.adapter.weight.grad, rtol=2e-5, atol=2e-6)


def test_simvp_matches_official_first_future_frame() -> None:
    torch.manual_seed(11)
    model = build_openstl("simvp", TINY_OPTIONS["simvp"] | {"image_size": 16})
    namespace = {"torch": torch, "nn": nn, "trunc_normal_": nn.init.trunc_normal_}
    official_definitions("simvp_modules.py", {"BasicConv2d", "ConvSC", "GroupConv2d", "gInception_ST"}, namespace)
    official_definitions(
        "simvp_model.py", {"SimVP_Model", "sampling_generator", "Encoder", "Decoder", "MidIncepNet"}, namespace
    )
    reference = namespace["SimVP_Model"]((7, 12, 16, 16), hid_S=8, hid_T=16, N_S=2, N_T=2, model_type="IncepU")
    reference.load_state_dict(model.state_dict(), strict=True)
    history = torch.rand(2, 12, 7, 16, 16)
    expected = reference(history.permute(0, 2, 1, 3, 4).contiguous())[:, :1].permute(0, 2, 1, 3, 4)
    prediction, auxiliary = model(history)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    assert auxiliary == {}


def test_patch_round_trip_keeps_channel_and_pixel_order() -> None:
    history = torch.arange(2 * 12 * 7 * 16 * 16).reshape(2, 12, 7, 16, 16)
    patched = _patchify(history, 4)
    reconstructed = torch.cat([_unpatchify(frame, 4, 12) for frame in patched.unbind(1)], dim=2)
    torch.testing.assert_close(reconstructed, history)


def test_defaults_match_parameter_budget() -> None:
    expected = {"convlstm": 44_115_120, "predrnn_v2": 44_110_752, "simvp": 43_830_156}
    for method, count in expected.items():
        with torch.device("meta"):
            model = build_openstl(method, {})
        assert sum(parameter.numel() for parameter in model.parameters()) == count
        assert abs(count / 44_162_136 - 1) < 0.01


def test_upstream_copies_have_recorded_hashes() -> None:
    manifest = json.loads((PORT_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
    copies = manifest["reference_files"] | {
        "convlstm_cell.py": "openstl/modules/convlstm_modules.py",
        "predrnnv2_cell.py": "openstl/modules/predrnnv2_modules.py",
        "LICENSE": "LICENSE",
    }
    for local, original in copies.items():
        assert hashlib.sha256((PORT_ROOT / local).read_bytes()).hexdigest() == manifest["source_sha256"][original]


@pytest.mark.parametrize("method", TINY_OPTIONS)
def test_rejects_wrong_context_length(method: str) -> None:
    model = build_openstl(method, TINY_OPTIONS[method] | {"image_size": 16})
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, 12, 4, 16, 16))
