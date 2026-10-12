import torch

from src.residual_diffusion_stage.diagnose_ccc_false_alarms import origin_masks


def test_disjoint_origin_partitions_and_previous_false_alarm() -> None:
    history = torch.zeros(1, 1, 2, 24, 24)
    history[0, 0, -1, 0, 0] = 0.1
    history[0, 0, 0, 0, 1] = 0.1
    pred = torch.zeros(1, 1, 2, 24, 24)
    pred[0, 0, :, 0, :3] = 0.02
    pred[0, 0, :, 23, 23] = 0.02
    truth = torch.zeros_like(pred)
    rows = origin_masks(pred, truth, history, 0.01)
    for row in rows:
        historical = sum(row[k].int() for k in ("last_observed_wet", "earlier_history_wet_only", "all_history_dry_near_8px", "all_history_dry_far_8px"))
        temporal = sum(row[k].int() for k in ("previous_truth_wet", "persistent_previous_false_alarm", "new_from_previous_pred_dry"))
        assert torch.equal(historical, row["false_pixels"].int())
        assert torch.equal(temporal, row["false_pixels"].int())
        assert int(row["all_history_dry_far_8px"].sum()) == 1
    assert int(rows[0]["previous_truth_wet"].sum()) == 1
    assert int(rows[1]["persistent_previous_false_alarm"].sum()) == 4
    assert int(rows[1]["weak_under_0.03"].sum()) == 4
