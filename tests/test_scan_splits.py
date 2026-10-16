from scripts.preprocess_luna16.convert_process_to_task03 import _assign_scan_splits
import pytest

from luna16_data import build_slice_lookup, neighbor_indices


def test_scan_split_is_deterministic_and_stratified_by_subset():
    rows = [
        {
            "case_name": f"nodule_{index}",
            "source_series_index": str(index),
            "seriesuid": f"uid-{index}",
            "subset": str(index // 10),
            "slice_count": 1,
        }
        for index in range(20)
    ]

    first = _assign_scan_splits([dict(row) for row in rows], seed=42)
    second = _assign_scan_splits([dict(row) for row in rows], seed=42)

    assert [row["split"] for row in first] == [row["split"] for row in second]
    for subset in ("0", "1"):
        splits = [row["split"] for row in first if row["subset"] == subset]
        assert splits.count("test") == 1
        assert splits.count("validation") == 1
        assert splits.count("train") == 8


def test_25d_neighbors_follow_numeric_slice_ids_after_training_shuffle():
    paths = [
        "/data/nodule_0/images/0_0002.png",
        "/data/nodule_1/images/1_0000.png",
        "/data/nodule_0/images/0_0000.png",
        "/data/nodule_0/images/0_0001.png",
    ]
    lookup = build_slice_lookup(paths)
    assert neighbor_indices(3, paths, lookup) == (2, 3, 0)
    assert neighbor_indices(2, paths, lookup) == (2, 2, 3)
    assert neighbor_indices(1, paths, lookup) == (1, 1, 1)


def test_25d_rejects_missing_internal_slice():
    paths = [
        "/data/nodule_0/images/0_0000.png",
        "/data/nodule_0/images/0_0002.png",
    ]
    with pytest.raises(ValueError, match="Missing 2.5D slice index"):
        build_slice_lookup(paths)
