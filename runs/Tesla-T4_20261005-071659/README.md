# Colab Tesla T4 run (5 Oct 2026)

The notebook as it stood on 5 Oct 2026, run on Google Colab with a Tesla T4 (PyTorch 2.11, CUDA 13.0). It is kept as a historical record, so its files are unchanged.

**Old metric names.** This run predates the four-denominator MFU reporting, so `results/metrics.json` uses the old keys. The T4 has no native bf16, so its low-precision runs used **fp16**, but the keys still say `bf16`:

| key in this run's `metrics.json` | means | new name in the main notebook |
|---|---|---|
| `mfu.peak_fp32_theory` | fp32 theoretical (published spec, Tesla T4), 8.1 TFLOP/s | `peaks.fp32_theoretical` |
| `mfu.peak_fp32_measured` | fp32 measured (8192² matmul, Tesla T4) | `peaks.fp32_measured` |
| `mfu.peak_bf16_theory` | **fp16** theoretical (published spec, Tesla T4), 65 TFLOP/s | `peaks.lowp_theoretical` |
| `mfu.peak_bf16_measured` | **fp16** measured (8192² matmul, Tesla T4) | `peaks.lowp_measured` |
| `mfu.bf16_achieved`, `mfu.bf16_mfu_*` | the **fp16** autocast run | `achieved_flops_per_s.lowp_run`, `lowp_run` |
| `mfu.*_mfu_vs_fp32`, `mfu.*_mfu_vs_bf16` | ratios against the old max(theoretical, measured) "peak" | replaced by all four named denominators |
| `mfu_sweep[].bf16` | fp16 run's FLOP/s at each width | `mfu_sweep[].lowp_run_flops` |

The main README's T4 numbers are computed from these saved values, divided by each of the four denominators. `figures/mfu_sweep_relabelled.png` is re-plotted from the same saved numbers with the new labels. `figures/mfu_sweep.png` and `training_loop_executed.ipynb` are the run's original outputs, with the old labels.
