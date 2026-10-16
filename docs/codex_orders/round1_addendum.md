# Work order R1 — addendum (on-device notes)

（這則補充來自 Claude，協調者。）

These points override `round1_cross_slice_attention.md` where they differ.

## A1. Neighbours by numeric slice id, never by list position

Commit `85f6354` on this branch already replaced list-position neighbours with
`luna16_data.build_slice_lookup` / `neighbor_indices`, because the training
list is shuffled before the dataset is built (the old rule gave wrong
z-neighbours for every training slice). The existing checkpoints we compare
against were trained with this fix.

For K slices, generalise **this** lookup: for center slice id `z`, use ids
`z-r .. z+r` (r = K//2) in the same case directory; a missing id at the case
border is replaced by the nearest valid id on that side (i.e. clamp toward the
center). Do this in one shared helper used by the dataset, `eval_common`,
`eval_lesion.py`, and the lesion-CPM validation. With K=3 the result must be
identical to the current `neighbor_indices`. Keep the existing tests in
`tests/test_scan_splits.py` passing and add a K=5 border test with a shuffled
path list.

## A2. Environment

- Edit code in this Windows clone of the branch. The Python env with torch
  lives inside WSL and your sandbox cannot reach WSL, so you cannot run the
  real test suite. Claude runs `ALLOW_RANDOM_BACKBONE=1 python -m pytest -q`
  in WSL after each of your turns and sends you the output. (If a Windows
  `python` with numpy is available you may use it for quick syntax checks,
  e.g. `python -m py_compile`.)
- There is no data in this clone; do not create `data/`, `checkpoints/`,
  `results/` or `logs/`. Do not commit or push; Claude reviews and commits.
- Tests must not need the real data or GPU (CPU, tiny synthetic inputs).

## A3. Final reply

List the files you changed/added, the test command output, and any point
where you deviated from the order and why.
