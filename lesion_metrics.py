"""Lesion-level (per-nodule) detection metrics for stacked 2.5D predictions.

Clinical reading asks whether each nodule was found, not how many pixels of a
pseudo-mask were covered. These helpers rebuild every test scan as a 3D volume,
treat each 3D connected ground-truth component as one nodule, and turn each 3D
connected predicted component into one candidate finding.

Hit rule: a candidate detects a nodule when the two components share at least
one voxel. A nodule detected by several candidates counts once and its extra
candidates are not false positives (mirrors the LUNA16 rule of ignoring
duplicate hits). Every candidate touching no nodule is a false positive.

A candidate's confidence is the maximum probability inside it, so sweeping a
confidence threshold yields a FROC curve: sensitivity versus average false
positives per scan. The competition performance metric (CPM) is the mean
sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


FROC_FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


@dataclass
class CaseResult:
    """Candidate list for one scan."""

    case_id: str
    num_nodules: int
    # One row per candidate: (confidence, index of the nodule it hits or -1).
    candidates: list = field(default_factory=list)
    nodule_voxels: list = field(default_factory=list)


def label_components(volume):
    labels, count = ndimage.label(np.asarray(volume, dtype=bool), structure=_STRUCTURE_26)
    return labels, count


def evaluate_case(case_id, gt_volume, prob_volume, candidate_threshold=0.2, valid_volume=None):
    """Return the candidate list of one scan.

    gt_volume and prob_volume are (Z, H, W) arrays in slice order. valid_volume,
    when given, marks voxels that count; ignored voxels are removed from both
    the ground truth and the candidates.
    """
    gt = np.asarray(gt_volume) > 0.5
    prob = np.asarray(prob_volume, dtype=np.float32)
    if gt.shape != prob.shape:
        raise ValueError(f"Shape mismatch for {case_id}: gt {gt.shape} vs prob {prob.shape}")
    if valid_volume is not None:
        valid = np.asarray(valid_volume) > 0.5
        gt = gt & valid
        prob = np.where(valid, prob, 0.0)

    gt_labels, num_nodules = label_components(gt)
    pred_labels, num_candidates = label_components(prob >= candidate_threshold)
    result = CaseResult(case_id=case_id, num_nodules=int(num_nodules))
    if num_nodules:
        result.nodule_voxels = [int(v) for v in np.bincount(gt_labels.ravel())[1:]]
    if num_candidates == 0:
        return result

    index = np.arange(1, num_candidates + 1)
    scores = ndimage.maximum(prob, labels=pred_labels, index=index)
    for candidate_id, score in zip(index, np.atleast_1d(scores)):
        overlap = gt_labels[pred_labels == candidate_id]
        overlap = overlap[overlap > 0]
        if overlap.size:
            # The nodule sharing most voxels with the candidate is the one it hits.
            hit = int(np.bincount(overlap).argmax()) - 1
        else:
            hit = -1
        result.candidates.append((float(score), hit))
    return result


def _sweep(case_results):
    """Sensitivity and FP/scan for every distinct confidence threshold, high to low."""
    num_scans = len(case_results)
    total_nodules = sum(case.num_nodules for case in case_results)
    rows = []
    for case_index, case in enumerate(case_results):
        for score, hit in case.candidates:
            rows.append((score, case_index, hit))
    rows.sort(key=lambda row: -row[0])

    detected = set()
    false_positives = 0
    sensitivities, fp_per_scan, thresholds = [], [], []
    position = 0
    while position < len(rows):
        threshold = rows[position][0]
        while position < len(rows) and rows[position][0] == threshold:
            _, case_index, hit = rows[position]
            if hit >= 0:
                detected.add((case_index, hit))
            else:
                false_positives += 1
            position += 1
        thresholds.append(threshold)
        sensitivities.append(len(detected) / total_nodules if total_nodules else 0.0)
        fp_per_scan.append(false_positives / num_scans if num_scans else 0.0)
    return np.asarray(thresholds), np.asarray(sensitivities), np.asarray(fp_per_scan), total_nodules


def froc(case_results, fp_rates=FROC_FP_RATES):
    """Sensitivity at fixed average false positives per scan, plus CPM.

    For each target rate the reported sensitivity is the best one reachable
    without exceeding that many false positives per scan.
    """
    thresholds, sensitivities, fp_per_scan, total_nodules = _sweep(case_results)
    at_rate = {}
    for rate in fp_rates:
        allowed = fp_per_scan <= rate + 1e-12
        at_rate[rate] = float(sensitivities[allowed].max()) if allowed.any() else 0.0
    return {
        "num_scans": len(case_results),
        "num_nodules": int(total_nodules),
        "sensitivity_at_fp_per_scan": {str(rate): value for rate, value in at_rate.items()},
        "cpm": float(np.mean(list(at_rate.values()))) if at_rate else 0.0,
        "max_sensitivity": float(sensitivities.max()) if sensitivities.size else 0.0,
        "fp_per_scan_at_max_sensitivity": float(fp_per_scan[-1]) if fp_per_scan.size else 0.0,
        "curve": {
            "thresholds": thresholds.tolist(),
            "sensitivity": sensitivities.tolist(),
            "fp_per_scan": fp_per_scan.tolist(),
        },
    }


def operating_point(case_results, threshold=0.5):
    """Sensitivity, FP/scan and lesion precision for candidates scoring >= threshold."""
    total_nodules = sum(case.num_nodules for case in case_results)
    detected = 0
    false_positives = 0
    true_candidates = 0
    for case in case_results:
        hits = set()
        for score, hit in case.candidates:
            if score < threshold:
                continue
            if hit >= 0:
                hits.add(hit)
                true_candidates += 1
            else:
                false_positives += 1
        detected += len(hits)
    num_scans = len(case_results)
    kept = true_candidates + false_positives
    return {
        "threshold": threshold,
        "sensitivity": detected / total_nodules if total_nodules else 0.0,
        "detected_nodules": detected,
        "num_nodules": total_nodules,
        "false_positives": false_positives,
        "fp_per_scan": false_positives / num_scans if num_scans else 0.0,
        "lesion_precision": true_candidates / kept if kept else 1.0,
    }


def sensitivity_by_size(case_results, threshold=0.5, bins=((0, 30), (30, 100), (100, 300), (300, None))):
    """Sensitivity grouped by nodule volume in voxels (small nodules are the clinical risk)."""
    groups = {}
    for case in case_results:
        hits = {hit for score, hit in case.candidates if score >= threshold and hit >= 0}
        for nodule_index, voxels in enumerate(case.nodule_voxels):
            for low, high in bins:
                if voxels >= low and (high is None or voxels < high):
                    key = f"{low}-{high if high is not None else 'inf'}"
                    total, found = groups.get(key, (0, 0))
                    groups[key] = (total + 1, found + (nodule_index in hits))
                    break
    return {
        key: {"nodules": total, "detected": found, "sensitivity": found / total if total else 0.0}
        for key, (total, found) in groups.items()
    }


class CaseStreamAccumulator:
    """Collect slice predictions arriving in case order and score each case when it ends.

    Slices must arrive grouped by case and in slice order, which is how
    load_data orders every split. Only the current case is held in memory.
    """

    def __init__(self, candidate_threshold=0.2):
        self.candidate_threshold = candidate_threshold
        self.results = []
        self._case_id = None
        self._gt, self._prob, self._valid = [], [], []

    def add(self, case_id, gt_slice, prob_slice, valid_slice=None):
        if case_id != self._case_id:
            self.finish()
            self._case_id = case_id
        self._gt.append(np.asarray(gt_slice) > 0.5)
        self._prob.append(np.asarray(prob_slice, dtype=np.float16))
        self._valid.append(None if valid_slice is None else np.asarray(valid_slice) > 0.5)

    def finish(self):
        if self._case_id is None:
            return
        if any(v is None for v in self._valid):
            valid = None
        else:
            valid = np.stack(self._valid)
        self.results.append(
            evaluate_case(
                self._case_id,
                np.stack(self._gt),
                np.stack(self._prob).astype(np.float32),
                candidate_threshold=self.candidate_threshold,
                valid_volume=valid,
            )
        )
        self._case_id = None
        self._gt, self._prob, self._valid = [], [], []

    def summary(self, operating_threshold=0.5):
        self.finish()
        report = froc(self.results)
        report["operating_point"] = operating_point(self.results, operating_threshold)
        report["sensitivity_by_size_voxels"] = sensitivity_by_size(self.results, operating_threshold)
        report["candidate_threshold"] = self.candidate_threshold
        return report
