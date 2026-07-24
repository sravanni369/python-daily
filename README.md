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
| 07 | day07_autograd_training_loop.py | PyTorch autograd: manual linear-regression training loop (forward, MSE, backward(), hand-written SGD, grad zeroing, un-standardizing learned weights) — no nn.Module, no optimizer (uses torch) |
| 08 | day08_csv_json_wrangler.py | CSV to JSON wrangler: best-effort type coercion (int/float/string/None), required-field validation that skips bad rows with line-numbered errors instead of crashing, stable-key JSON output with round-trip check |
| 09 | day09_forecast_table.py | National forecast table parser: regex tokenizing of packed "hi/lo/sky" cells with a code legend, dataclasses for today vs next-day, rejects malformed rows instead of crashing, and reports hottest/coolest/widest-swing cities and sky distribution |

