# python-daily

One focused Python exercise per day. Each file is self-contained and runnable: `python dayNN_topic.py`. Standard library only unless noted.

## Progress

| Day | File | What it shows |
|-----|------|---------------|
| 01 | day01_data_quality_checker.py | Column-level CSV quality checks: nulls, duplicate keys, numeric ranges, regex formats |
| 02 | day02_reconcile_datasets.py | Source-vs-target reconciliation: key set diff, duplicate keys, field-level mismatches, match rate |
| 03 | day03_log_parser.py | Web-server log parsing with one named-group regex: request counts, status distribution, error rate, slowest endpoints, top talkers |
| 04 | day04_rag_chunker.py | Overlapping text chunking for RAG: fixed word-size windows with overlap, splitting on paragraph/sentence boundaries so ideas stay whole |
| 05 | day05_pytorch_tensors.py | PyTorch tensor fundamentals: creation (rand/tensor/ones/zeros), rank & shape, indexing, element-wise math, .item(), NumPy bridge, device selection, and a first look at autograd (uses torch + numpy) |
| 06 | day06_sql_with_sqlite.py | SQL via sqlite3: INNER/LEFT joins, GROUP BY/HAVING, and data-validation queries (orphan records, duplicates, business-rule checks) on a tiny claims DB |
| 07 | day07_autograd_training_loop.py | PyTorch autograd: manual linear-regression training loop (forward, MSE, backward(), hand-written SGD, grad zeroing, un-standardizing learned weights). No nn.Module, no optimizer (uses torch) |
| 08 | day08_csv_json_wrangler.py | CSV to JSON wrangler: best-effort type coercion (int/float/string/None), required-field validation that skips bad rows with line-numbered errors instead of crashing, stable-key JSON output with round-trip check |
| 09 | day09_forecast_table.py | National forecast table parser: regex tokenizing of packed "hi/lo/sky" cells with a code legend, dataclasses for today vs next-day, rejects malformed rows instead of crashing, and reports hottest/coolest/widest-swing cities and sky distribution |
| 10 | day10_eval_metrics.py | Evaluation metrics from scratch: precision/recall/F1 with zero-division handling, confusion matrix, macro averaging, sklearn-style report, and a spam-filter demo showing why accuracy flatters lazy models on imbalanced data |
| 11 | day11_one_hot_encoding.py | One-hot encoding from scratch (Low-Code AI, Stripling & Abel): why ordinal codes invent a false magnitude, stable category ordering between train and serve, and unseen categories encoding as all-zeros instead of crashing |
| 12 | day12_mode_imputation.py | Mode imputation for missing categorical values (Low-Code AI, ch. 2): deletion vs imputation, the many spellings of "missing" in real CSVs, deterministic alphabetical tie-breaking, and returning None when a column is entirely missing |
| 13 | day13_feature_scaling.py | Min-max scaling and its two failure modes (Low-Code AI, Stripling & Abel, p. 24): one outlier crushing every real value to 0, a constant column dividing by zero, and median/IQR robust scaling as the alternative |
| 14 | day14_musk_token_economics.py | Tokenizer byte economics (Hands-On LLM, pp. 74/78/82): GPT-style byte fallback means a brand name in Telugu or Hindi bills 3-6x its English cost; measures Musk company names across 6 market scripts, with xAI as the 1.00x control |
| 15 | day15_gil_threads_vs_processes.py | The global interpreter lock measured (Dive into Deep Learning, p. 270): the same CPU-bound work run serially, on threads and on processes, where threads give no speedup at all (0.90x to 1.04x across four runs) while processes give 2.08x, then the same three ways on I/O-bound work where the same threads give 3.97x. verify_day15.py re-times everything independently and asserts both claims. Same tool, opposite result, depending on what the work is waiting for |
| 16 | day16_validation_selection_bias.py | The third dataset (Low-Code AI, Stripling & Abel, PDF pp. 268-269): pick the best of 60 candidate models on one validation set and the winner scores 0.606 there but 0.498 on the untouched test set, where the labels are coin flips and 0.500 is the ceiling. Repeated 200 times because a single test set is noisy enough to hide it: validation flattered in 200/200 runs, by +0.107 on average. Rerun on data with real signal and the gap falls to +0.011 |
