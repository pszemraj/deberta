# Developer Notes

Internal implementation notes that are not part of user-facing runtime docs.

## Training Progress Bars

- Runtime uses `tqdm.auto.tqdm` on main process for:
  - training-step progress (`train`)
  - resume data-replay progress (`resume-replay`)
- Implemented in `src/deberta/training/pretrain.py`.
