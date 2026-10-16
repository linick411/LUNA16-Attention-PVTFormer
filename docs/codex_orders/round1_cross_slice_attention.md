# Work order R1: Cross-Slice Attention for lesion-level recall

（這則工作單來自 Claude，協調者。Codex 負責撰寫主程式；Claude 負責測試與驗證。）

Branch: `recall-cross-slice-attention` (already contains `lesion_metrics.py`
and `tests/test_lesion_metrics.py`, written by Claude as the verification
harness; do not change their behaviour).

## Goal

Raise **lesion-level sensitivity** (per-nodule recall, see `lesion_metrics.py`)
on the LUNA16 `sphere-v1` test split to 90%+, with the gain attributable to an
attention change. The fair comparison is:

| Arm | Model | Difference |
| --- | --- | --- |
| Control | `PVTFormerConcat25D` with K slices | none |
| Treatment | same control + Cross-Slice Attention (CSA) on the center slice | only the attention module |

Everything else (K, data, split, augmentation, loss, optimizer, LR schedule,
epochs, early stopping, checkpoint-selection rule, seed) must be identical.

## Tasks

### 1. K-slice input (default K stays 3, so old checkpoints still load)

- `luna16_data.py`: `LUNA16SliceDataset(..., num_slices=3)`; neighbours are
  `index-r .. index+r` (r = K//2), clamped to the same case directory
  (repeat the nearest valid slice at case borders). Output shape `(K, 1, H, W)`.
- `train_common.TrainingConfig` / `eval_common.EvaluationConfig`: field
  `num_slices = int(os.environ.get("NUM_SLICES", "3"))`, passed through.
  Must be odd and >= 1; raise `ValueError` otherwise.
- `eval_common._neighbor_indices` generalised the same way.
- `model_concat_25d.PVTFormerConcat25D(num_slices=3)`: the 1x1 fusion convs take
  `C * num_slices`; forward validates `inputs.shape[1] == num_slices`.

### 2. Cross-Slice Attention module — new file `model5.py`

```text
class CrossSliceAttention(nn.Module):
    def __init__(self, channels, num_slices, key_dim, heads=4, window=3)
    def forward(self, feats)  # feats (B, S, C, H, W) -> (center_out (B, C, H, W), attn (B, heads, H*W, S*window*window))
```

- Query from the center slice: `q = Conv1x1(C -> key_dim)(feats[:, S//2])`.
- Keys from all S slices: `Conv1x1(C -> key_dim)` + learnable slice-offset
  embedding `nn.Parameter(torch.zeros(S, key_dim))` added per slice.
- Values from all S slices: `Conv1x1(C -> C)`.
- Local window: each query position attends to a `window x window`
  neighbourhood around the same (y, x) on **every** slice
  (`F.unfold(kernel_size=window, padding=window//2)`), i.e. `S*window*window`
  tokens per query, multi-head scaled dot-product, softmax over tokens.
  Zero-padded positions must be masked out (−inf) before softmax.
- Output: `center + gamma * proj(attn_out)` with `proj = Conv1x1(C -> C)` and
  `gamma = nn.Parameter(torch.zeros(1))` (identity at initialisation).
- Must run under AMP fp16 without NaN (compute softmax in fp32).

Clinical rationale (for the docstring): a vessel continues as a tube through
many slices while a nodule is a compact blob that appears in a few slices;
letting each center-slice position compare itself with the same neighbourhood
on neighbouring slices lets the network separate the two, mirroring how a
radiologist scrolls through slices.

`PVTFormerCrossSliceAttention(num_slices=5)` in `model5.py`:
- identical to `PVTFormerConcat25D` (same backbone, fusion convs, decoder,
  heads) **plus** CSA applied at stages e1 (C=64, key_dim=32), e2 (C=128,
  key_dim=32) and e3 (C=320, key_dim=64). The CSA-enhanced center feature
  replaces the center slice's feature before the 1x1 concat fusion.
- `forward(inputs, return_attention=False)`; with `True` also return a dict of
  the three attention maps (for visualisation in the report).

### 3. Lesion-level evaluation entry point — new file `eval_lesion.py`

```text
python eval_lesion.py --model {baseline,concat_25d,attention_gate,voxel_attention,coordinate_attention,cross_slice_attention}
                      [--checkpoint PATH] [--num-slices K] [--split test|validation]
                      [--candidate-threshold 0.2] [--operating-threshold 0.5]
                      [--output-dir results/lesion/<model>] [--batch-size 16]
```

- Batched inference over the split in `load_data` order (cases are contiguous),
  feeding each slice's probability map into
  `lesion_metrics.CaseStreamAccumulator` with case id = parent-of-parent dir
  name (e.g. `nodule_123`). Honour `ignore/` masks like `eval_common` does.
- Also accumulate the existing pixel micro metrics (`metrics.confusion_counts`)
  so one run gives both views.
- Write `lesion_metrics.json` (summary incl. FROC curve, CPM, operating point,
  size-binned sensitivity, pixel micro metrics, model/checkpoint/num_slices/
  device/torch version) and `froc.csv` (fp_per_scan, sensitivity, threshold).
- Models with `num_slices == 1` (baseline) use the existing 2D RGB input path.

### 4. Training entry points

- `train_cross_slice_attention.py` / `eval_cross_slice_attention.py`
  (mirroring the coordinate-attention scripts; checkpoint
  `checkpoint_cross_slice_attention.pth`).
- `train_common`: optional checkpoint selection by validation lesion score.
  `SELECT_METRIC` env: `f1` (default, current behaviour) or `lesion_cpm`.
  With `lesion_cpm`, after each validation epoch compute the CPM on the
  validation split via `CaseStreamAccumulator` (the valid loader is not
  shuffled, so slice order is case order; carry the case id per sample, e.g.
  make the dataset optionally return it or derive from `valid_x`). Log
  `valid_lesion_cpm` and `valid_lesion_sens@0.5` to the history CSV (new
  columns appended at the end).
- Register `cross_slice_attention` in `scripts/run_experiments.py`.

### 5. Tests (pytest, CPU, `ALLOW_RANDOM_BACKBONE=1`, tiny input like 64x64)

- CSA: output shape, gamma=0 ⇒ output equals center input exactly,
  attention rows sum to 1, padded tokens get zero weight, gradients reach
  q/k/v after one optimizer step that moves gamma.
- `PVTFormerCrossSliceAttention(num_slices=5)` forward/backward on (2,5,1,64,64).
- `PVTFormerConcat25D(num_slices=5)` forward; default K=3 state-dict keys
  unchanged versus current main.
- K-slice dataset neighbour clamping at case borders.
- `eval_lesion` on a tiny synthetic Task03 folder (2 cases, a few PNG slices)
  with a random-backbone model checkpoint runs end to end and writes both files.
- Whole suite `python -m pytest -q` passes.

## Out of scope

No change to the loss, augmentation, optimizer, or the existing models'
default behaviour. Do not commit data, checkpoints, or results directories.
