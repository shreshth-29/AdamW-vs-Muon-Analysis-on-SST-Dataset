# AdamW vs Muon: Diagnosing Optimization during Fine-Tuning

DAPLab recruitment task (post-training optimization). Fine-tuning a small LM on SST-2 with AdamW and with Muon, and diagnosing how the two optimizers differ — not just in accuracy, but in the trajectory they take and the solution they land in.

**Report:** [`AdamW_vs_Muon_report.pdf`](AdamW_vs_Muon_report.pdf) (one page: setup, metrics, results table, and answers to the four task questions).


## Files

- `train_multi_config.ipynb` — config-driven training. Each entry in `configs` is one run; saves `ckpt/<run>/logs.json` and weights at steps 0, 500, 501 and the end.
- `final_analysis.ipynb` — computes the metrics from the checkpoints, caches them in `stats.json`, and makes the report figure and tables.
- `AdamW_vs_Muon_report.pdf` — the write-up.


## Running

```
pip install torch>=2.9 transformers datasets matplotlib tqdm
```

Run all cells of `train.ipynb` (it skips any run whose `logs.json` already exists), then `analysis.ipynb`. Works on CUDA or Apple MPS; the results in the report were produced on an MPS Device (locally trained on M5 MacBook).




