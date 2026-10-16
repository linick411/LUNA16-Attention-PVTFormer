import numpy as np
import pytest

from lesion_metrics import (
    CaseStreamAccumulator,
    evaluate_case,
    froc,
    operating_point,
    sensitivity_by_size,
)


def _ball(shape, center, radius):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    return (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2


def test_detected_and_missed_nodules_and_false_positive():
    shape = (12, 32, 32)
    gt = _ball(shape, (3, 8, 8), 2) | _ball(shape, (8, 24, 24), 2)
    prob = np.zeros(shape, dtype=np.float32)
    prob[_ball(shape, (3, 8, 8), 1)] = 0.9  # hits nodule 1
    prob[_ball(shape, (8, 8, 24), 1)] = 0.7  # false positive far from both

    case = evaluate_case("scan", gt, prob, candidate_threshold=0.2)
    assert case.num_nodules == 2
    assert sorted(case.candidates) == [(pytest.approx(0.7), -1), (pytest.approx(0.9), 0)]

    point = operating_point([case], threshold=0.5)
    assert point["sensitivity"] == 0.5
    assert point["false_positives"] == 1
    assert point["fp_per_scan"] == 1.0
    assert point["lesion_precision"] == 0.5


def test_duplicate_hits_count_once_and_are_not_false_positives():
    shape = (6, 20, 20)
    gt = _ball(shape, (3, 10, 10), 3)
    prob = np.zeros(shape, dtype=np.float32)
    prob[3, 8, 8] = 0.8
    prob[3, 12, 12] = 0.6  # second disconnected candidate on the same nodule

    case = evaluate_case("scan", gt, prob)
    point = operating_point([case])
    assert point["detected_nodules"] == 1
    assert point["false_positives"] == 0


def test_froc_reports_best_sensitivity_within_fp_budget():
    shape = (6, 20, 20)
    cases = []
    for index in range(4):
        gt = _ball(shape, (3, 10, 10), 2)
        prob = np.zeros(shape, dtype=np.float32)
        # Nodule found with confidence that drops case by case.
        prob[3, 10, 10] = 0.9 - 0.2 * index
        # One confident false positive in the first scan only.
        if index == 0:
            prob[1, 2, 2] = 0.95
        cases.append(evaluate_case(f"scan{index}", gt, prob, candidate_threshold=0.05))

    report = froc(cases)
    assert report["num_nodules"] == 4
    assert report["num_scans"] == 4
    # Zero FP budget: the 0.95 FP comes first, so nothing is detected before it.
    assert report["sensitivity_at_fp_per_scan"]["0.125"] == 0.0
    # 0.25 FP/scan allows that single false positive -> all four nodules.
    assert report["sensitivity_at_fp_per_scan"]["0.25"] == 1.0
    assert report["max_sensitivity"] == 1.0


def test_ignore_mask_removes_voxels_from_truth_and_candidates():
    shape = (4, 10, 10)
    gt = np.zeros(shape, dtype=bool)
    gt[2, 2, 2] = True
    prob = np.zeros(shape, dtype=np.float32)
    prob[2, 7, 7] = 0.9
    valid = np.ones(shape, dtype=bool)
    valid[2, 2, 2] = False
    valid[2, 7, 7] = False

    case = evaluate_case("scan", gt, prob, valid_volume=valid)
    assert case.num_nodules == 0
    assert case.candidates == []


def test_stream_accumulator_splits_cases_in_order():
    shape = (5, 16, 16)
    acc = CaseStreamAccumulator(candidate_threshold=0.3)
    for case_id, hit in (("a", True), ("b", False)):
        gt = _ball(shape, (2, 8, 8), 2)
        prob = np.where(gt, 0.9 if hit else 0.0, 0.0).astype(np.float32)
        for z in range(shape[0]):
            acc.add(case_id, gt[z], prob[z])
    report = acc.summary()
    assert [case.case_id for case in acc.results] == ["a", "b"]
    assert report["operating_point"]["sensitivity"] == 0.5
    assert report["num_scans"] == 2


def test_sensitivity_by_size_groups_nodules():
    shape = (8, 30, 30)
    gt = _ball(shape, (4, 8, 8), 1) | _ball(shape, (4, 20, 20), 3)
    prob = np.zeros(shape, dtype=np.float32)
    prob[_ball(shape, (4, 20, 20), 3)] = 0.9
    case = evaluate_case("scan", gt, prob)
    groups = sensitivity_by_size([case])
    assert groups["0-30"] == {"nodules": 1, "detected": 0, "sensitivity": 0.0}
    assert groups["100-300"]["sensitivity"] == 1.0
