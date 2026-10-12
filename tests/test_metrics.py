import numpy as np
import pytest

from eval_full_metrics import evaluate_full
from eval_sap import greedy_tp, iou_matrix


def fixture(classes):
    n = len(classes)
    rec = {
        "box": np.array([[0, 0, 20, 20]] * n, dtype=np.float32),
        "score": np.linspace(0.99, 0.80, n),
        "cls": np.array(classes),
        "ptr": np.array([0, n]),
        "scene": np.array([0]),
        "frame_idx": np.array([0]),
    }
    gt = {"frame.jpg": (np.array([[0, 0, 20, 20]], dtype=np.float32), np.array([0]))}
    frames = [(0, 0, 0, "frame.jpg")]
    return rec, gt, frames, {(0, 0): 0}


def test_wrong_class_cannot_claim_ground_truth():
    result = evaluate_full(*fixture([1]))
    assert result["AP50"] == 0


def test_perfect_detection_and_duplicate_matching():
    result = evaluate_full(*fixture([0]))
    assert result["AP50"] == pytest.approx(1)
    iou = iou_matrix(np.array([[0, 0, 20, 20]] * 2), np.array([[0, 0, 20, 20]]))
    assert greedy_tp(iou, 0.5).tolist() == [True, False]


def test_missing_frame_is_rejected():
    rec, gt, frames, keys = fixture([0])
    rec["scene"] = np.array([99])
    with pytest.raises(SystemExit, match="时序表"):
        evaluate_full(rec, gt, frames, keys)


def test_empty_predictions_are_finite():
    rec, gt, frames, keys = fixture([0])
    rec["box"] = np.empty((0, 4), dtype=np.float32)
    rec["score"] = np.empty(0)
    rec["cls"] = np.empty(0, dtype=np.int64)
    rec["ptr"] = np.array([0, 0])
    result = evaluate_full(rec, gt, frames, keys)
    assert result["AP50"] == 0 and result["AR100"] == 0
    assert result["operating_point"]["F1"] == 0
    import json

    json.dumps(result, allow_nan=False)
